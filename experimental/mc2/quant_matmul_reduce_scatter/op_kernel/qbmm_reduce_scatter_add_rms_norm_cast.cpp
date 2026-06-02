/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 */

#include <cstdint>
#include <cstdlib>
#include <type_traits>
#include "block/block_mmad.h"
#include "block/block_scheduler_policy.h"
#include "lib/matmul_intf.h"
#include "qbmm_reduce_scatter_add_rms_norm_cast_tiling_data.h"
#include "include/tensor_api/tensor.h"
#include "kernel_operator.h"
#include "adv_api/hccl/hccl.h"
#include "adv_api/reduce/sum.h"
#include "adv_api/pad/broadcast.h"
#include "kernel_tiling/kernel_tiling.h"
#include "kernel/quant_matmul_mx_kernel_swat.h"
#include "kernel_utils/common_utils.h"
#include "block/quant_matmul_mx_block_mmad_swat.h"

using namespace AscendC;

__global__ __aicore__ void qbmm_reduce_scatter_add_rms_norm_cast(
    GM_ADDR x1, GM_ADDR x2, GM_ADDR y, GM_ADDR gamma,
    GM_ADDR dScaleB, GM_ADDR bias,  GM_ADDR dScaleA,
    GM_ADDR y1, GM_ADDR y2, GM_ADDR x,
    GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    REGISTER_TILING_DEFAULT(QbmmReduceScatterAddRmsNormCastTilingData);
    GET_TILING_DATA_WITH_STRUCT(QbmmReduceScatterAddRmsNormCastTilingData, tilingData, tilingGM);
    using TypeA = fp8_e4m3fn_t;
    using TypeB = fp8_e4m3fn_t;
    using TypeC = bfloat16_t;
    constexpr int32_t SCALE_C0 = 2;
    constexpr int32_t C0_SIZE = AscendC::AuxGetC0Size<TypeA>();

    

    if (TILING_KEY_IS(0)) {
        // ND layout: TransA=false, TransB=true
        // if(AscendC::GetBlockIdx() == 0) {
        //     printf("enter nd mode\n");
        // }
        using LayoutA = typename AscendC::Te::FrameLayoutFormat<AscendC::Te::NDExtLayoutPtn>;
        using LayoutB = typename AscendC::Te::FrameLayoutFormat<AscendC::Te::DNExtLayoutPtn>;
        using LayoutC = typename AscendC::Te::FrameLayoutFormat<AscendC::Te::NDExtLayoutPtn>;
        using LayoutScaleA = typename AscendC::Te::FrameLayoutFormat<AscendC::Te::ScaleANDLayoutPtn, AscendC::Std::Int<SCALE_C0>>;
        using LayoutScaleB = typename AscendC::Te::FrameLayoutFormat<AscendC::Te::ScaleBDNLayoutPtn, AscendC::Std::Int<SCALE_C0>>;

        using BlockScheduler = QuantMatmulMxSwatScheduler<NO_FULL_LOAD_MODE>;
        using DispatchPolicy = QuantMatmulMxMultiBlockWithSwat<NO_FULL_LOAD_MODE, 2UL>;
        using ProblemShape = MatmulShape;
        using TypeATuple = AscendC::Std::tuple<TypeA, fp8_e8m0_t>;
        using TypeBTuple = AscendC::Std::tuple<TypeB, fp8_e8m0_t>;
        using LayoutATuple = AscendC::Std::tuple<LayoutA, LayoutScaleA>;
        using LayoutBTuple = AscendC::Std::tuple<LayoutB, LayoutScaleB>;
        using BlockMmad = Block::BlockMmad<DispatchPolicy, TypeATuple, LayoutATuple, TypeBTuple, LayoutBTuple, TypeC, LayoutC>;
        using KernelImpl = Kernel::QuantMatmulMxKernelSwat<ProblemShape, BlockMmad, BlockScheduler>;
        using Params = typename KernelImpl::Params;
        using CommParams = typename KernelImpl::CommParams;
        using OutputParams = typename KernelImpl::OutputParams;

        ProblemShape problemShape{tilingData.qbmmTilingdata.m, tilingData.qbmmTilingdata.n, tilingData.qbmmTilingdata.k, 1UL};
        BlockMmad::Params mmadParams{x1, x2, x, dScaleA, dScaleB};
        KernelImpl::L1Params l1Params{
            static_cast<uint64_t>(tilingData.qbmmTilingdata.stepK) * tilingData.qbmmTilingdata.baseK,
            tilingData.qbmmTilingdata.scaleKL1};
        KernelImpl::BlockSchedulerParams schedulerParams{
            tilingData.qbmmTilingdata.baseM, tilingData.qbmmTilingdata.baseN,
            tilingData.qbmmTilingdata.mTailTile, tilingData.qbmmTilingdata.nTailTile,
            tilingData.qbmmTilingdata.mBaseTailSplitCnt, tilingData.qbmmTilingdata.nBaseTailSplitCnt,
            tilingData.qbmmTilingdata.mTailMain, tilingData.qbmmTilingdata.nTailMain};
        KernelImpl::QBMMTiling qbmmParams{
            tilingData.qbmmTilingdata.baseM, tilingData.qbmmTilingdata.baseN,
            tilingData.qbmmTilingdata.baseK, tilingData.qbmmTilingdata.dbL0c};
        CommParams commParams{
            problemShape.m / tilingData.qbmmReduceScatterAddRmsNormCastTilingInfo.rankSize,
            static_cast<int32_t>(tilingData.qbmmReduceScatterAddRmsNormCastTilingInfo.rankSize)};
        OutputParams outputParams{y1, y2, x, y, gamma};

        KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
        Params params{problemShape, mmadParams, l1Params, schedulerParams, qbmmParams, commParams, outputParams};
        KernelImpl kernel;
        kernel(params);
        return;
    }

    if (TILING_KEY_IS(1)) {
        // NZ layout: TransA=false, TransB=false, NZLayoutPtn
        // if(AscendC::GetBlockIdx() == 0) {
        //     printf("enter nz mode\n");
        // }
        using LayoutA = typename AscendC::Te::FrameLayoutFormat<AscendC::Te::NDExtLayoutPtn>;
        using LayoutB = typename AscendC::Te::FrameLayoutFormat<AscendC::Te::NZLayoutPtn, AscendC::Std::Int<C0_SIZE>>;
        using LayoutC = typename AscendC::Te::FrameLayoutFormat<AscendC::Te::NDExtLayoutPtn>;
        using LayoutScaleA = typename AscendC::Te::FrameLayoutFormat<AscendC::Te::ScaleANDLayoutPtn, AscendC::Std::Int<SCALE_C0>>;
        using LayoutScaleB = typename AscendC::Te::FrameLayoutFormat<AscendC::Te::ScaleBNDLayoutPtn, AscendC::Std::Int<SCALE_C0>>;

        using BlockScheduler = QuantMatmulMxSwatScheduler<NO_FULL_LOAD_MODE>;
        using DispatchPolicy = QuantMatmulMxMultiBlockWithSwat<NO_FULL_LOAD_MODE, 2UL>;
        using ProblemShape = MatmulShape;
        using TypeATuple = AscendC::Std::tuple<TypeA, fp8_e8m0_t>;
        using TypeBTuple = AscendC::Std::tuple<TypeB, fp8_e8m0_t>;
        using LayoutATuple = AscendC::Std::tuple<LayoutA, LayoutScaleA>;
        using LayoutBTuple = AscendC::Std::tuple<LayoutB, LayoutScaleB>;
        using BlockMmad = Block::BlockMmad<DispatchPolicy, TypeATuple, LayoutATuple, TypeBTuple, LayoutBTuple, TypeC, LayoutC>;
        using KernelImpl = Kernel::QuantMatmulMxKernelSwat<ProblemShape, BlockMmad, BlockScheduler>;
        using Params = typename KernelImpl::Params;
        using CommParams = typename KernelImpl::CommParams;
        using OutputParams = typename KernelImpl::OutputParams;

        ProblemShape problemShape{tilingData.qbmmTilingdata.m, tilingData.qbmmTilingdata.n, tilingData.qbmmTilingdata.k, 1UL};
        BlockMmad::Params mmadParams{x1, x2, workspaceGM, dScaleA, dScaleB};
        // BlockMmad::Params mmadParams{x1, x2, x, dScaleA, dScaleB};
        KernelImpl::L1Params l1Params{
            static_cast<uint64_t>(tilingData.qbmmTilingdata.stepK) * tilingData.qbmmTilingdata.baseK,
            tilingData.qbmmTilingdata.scaleKL1};
        KernelImpl::BlockSchedulerParams schedulerParams{
            tilingData.qbmmTilingdata.baseM, tilingData.qbmmTilingdata.baseN,
            tilingData.qbmmTilingdata.mTailTile, tilingData.qbmmTilingdata.nTailTile,
            tilingData.qbmmTilingdata.mBaseTailSplitCnt, tilingData.qbmmTilingdata.nBaseTailSplitCnt,
            tilingData.qbmmTilingdata.mTailMain, tilingData.qbmmTilingdata.nTailMain};
        KernelImpl::QBMMTiling qbmmParams{
            tilingData.qbmmTilingdata.baseM, tilingData.qbmmTilingdata.baseN,
            tilingData.qbmmTilingdata.baseK, tilingData.qbmmTilingdata.dbL0c};
        CommParams commParams{
            problemShape.m / tilingData.qbmmReduceScatterAddRmsNormCastTilingInfo.rankSize,
            static_cast<int32_t>(tilingData.qbmmReduceScatterAddRmsNormCastTilingInfo.rankSize)};
        OutputParams outputParams{y1, y2, x, y, gamma};

        KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
        Params params{problemShape, mmadParams, l1Params, schedulerParams, qbmmParams, commParams, outputParams};
        KernelImpl kernel;
        kernel(params);
        return;
    }
}
