/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 */

/*!
 * \file all_gather_matmul_impl.h
 * \brief AllGather + QuantizedMatmul 融合算子（1:2 CV 混核模式）
 */

#ifndef ALL_GATHER_MATMUL_IMPL_H
#define ALL_GATHER_MATMUL_IMPL_H

#include "basic_api/kernel_basic_intf.h"
#include "adv_api/hccl/hccl.h"
#include "kernel_tiling/kernel_tiling.h"
#include "all_gather_mte.h"
#include "all_gather_mte_comm.h"
#include "block/block_mmad.h"
#include "block/block_scheduler_policy.h"
#include "inc/kernel/moe_distribute_base.h"
#include "kernel/quant_matmul_mx_kernel_swat.h"
#include "kernel_utils/common_utils.h"
#include "block/quant_matmul_mx_block_mmad_swat.h"

namespace allGatherMatmulImpl {

#define TemplateMC2TypeClass typename X1Type, typename X2Type, typename ScaleType, uint32_t SyncMode
#define TemplateMC2TypeFunc X1Type, X2Type, ScaleType, SyncMode

using namespace AscendC;
using namespace AllGatherImpl;

constexpr static uint64_t WIN_ALIGN = 512;

template<TemplateMC2TypeClass>
class allGatherMatmulImpl {
public:
    __aicore__ inline allGatherMatmulImpl() {};

    __aicore__ inline void Init(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR x1Scale, GM_ADDR x2Scale,
                                GM_ADDR cGM, GM_ADDR allGatherDataOut, GM_ADDR allGatherScalesOut,
                                GM_ADDR tilingGM);
    __aicore__ inline void Process();

    __gm__ Mc2Kernel::HcclOpParam* context_;

private:
    __aicore__ inline void InitBaseParams(const __gm__ allGatherTilingData *tilingData);
    __aicore__ inline void AllGatherProcess();
    __aicore__ inline void MatmulProcess();

    TPipe *tpipe_{nullptr};
    AllGatherMte<int8_t, int8_t, int8_t, SyncMode, true> allGatherMte_;
    GM_ADDR allGatherDataOutAddr_;
    GM_ADDR allGatherScalesOutAddr_;
    GM_ADDR inputXAddr_;
    GM_ADDR inputXScaleAddr_;
    GM_ADDR selfRankAddr_;
    GM_ADDR scaleAAddr_;
    GM_ADDR bGM_;
    GM_ADDR cGM_;
    GM_ADDR x2ScaleGM;
    __gm__ allGatherMatmulTilingData* tilingData;

    uint32_t aivId_{0};
    uint32_t axisM_{0};
    uint32_t axisKa_{0};
    uint32_t axisN_{0};
    uint32_t rankSize_{0};
    uint64_t kL1_{0};

    GM_ADDR winContext_{nullptr};
};

template<TemplateMC2TypeClass>
__aicore__ inline void allGatherMatmulImpl<TemplateMC2TypeFunc>::Init(
    GM_ADDR aGM, GM_ADDR bGM, GM_ADDR x1Scale, GM_ADDR x2Scale,
    GM_ADDR cGM, GM_ADDR allGatherDataOut, GM_ADDR allGatherScalesOut,
    GM_ADDR tilingGM)
{
    tilingData = (__gm__ allGatherMatmulTilingData*)tilingGM;
    if ASCEND_IS_AIV {
        TPipe pipe;
        tpipe_ = &pipe;
    }
    winContext_ = AscendC::GetHcclContext<HCCL_GROUP_ID_0>();
    context_ = (__gm__ Mc2Kernel::HcclOpParam*)winContext_;

    InitBaseParams(&tilingData->allGatherTiling);

    allGatherDataOutAddr_ = allGatherDataOut;
    allGatherScalesOutAddr_ = allGatherScalesOut;
    inputXAddr_ = aGM;
    inputXScaleAddr_ = x1Scale;
    allGatherMte_.InitParams(&tilingData->allGatherTiling, winContext_);

    uint64_t xSize = axisM_ * axisKa_ * sizeof(int8_t);
    uint64_t winOffset = AscendC::CeilDiv(rankSize_ * xSize, WIN_ALIGN) * WIN_ALIGN;

    selfRankAddr_ = (GM_ADDR)allGatherMte_.GetSelfDataAddrGm();
    scaleAAddr_ = (GM_ADDR)allGatherMte_.GetSelfDataAddrGm() + winOffset;
    bGM_ = bGM;
    cGM_ = cGM;
    x2ScaleGM = x2Scale;

    if ASCEND_IS_AIC {
        return;
    }
}

template<TemplateMC2TypeClass>
__aicore__ inline void allGatherMatmulImpl<TemplateMC2TypeFunc>::InitBaseParams(const __gm__ allGatherTilingData *tilingData)
{
    aivId_ = GetBlockIdx();
    axisM_ = tilingData->M;
    axisKa_ = tilingData->Ka;
    axisN_ = tilingData->N;
    rankSize_ = tilingData->rankSize;
}

template<TemplateMC2TypeClass>
__aicore__ inline void allGatherMatmulImpl<TemplateMC2TypeFunc>::MatmulProcess()
{
    using TypeA = X1Type;
    using TypeB = X2Type;
    using TypeC = bfloat16_t;
    constexpr int32_t SCALE_C0 = 2;
    constexpr int32_t C0_SIZE = AscendC::AuxGetC0Size<TypeA>();
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

    ProblemShape problemShape{tilingData->qbmmTilingData.m, tilingData->qbmmTilingData.n,
        tilingData->qbmmTilingData.k, 1UL};
    typename BlockMmad::Params mmadParams{selfRankAddr_, bGM_, cGM_, scaleAAddr_, x2ScaleGM};
    typename KernelImpl::L1Params l1Params{
        static_cast<uint64_t>(tilingData->qbmmTilingData.stepK) * tilingData->qbmmTilingData.baseK,
        tilingData->qbmmTilingData.scaleKL1};
    kL1_ = l1Params.kL1;
    typename KernelImpl::BlockSchedulerParams schedulerParams{
        tilingData->qbmmTilingData.baseM,
        tilingData->qbmmTilingData.baseN,
        tilingData->qbmmTilingData.mTailTile,
        tilingData->qbmmTilingData.nTailTile,
        tilingData->qbmmTilingData.mBaseTailSplitCnt,
        tilingData->qbmmTilingData.nBaseTailSplitCnt,
        tilingData->qbmmTilingData.mTailMain,
        tilingData->qbmmTilingData.nTailMain};
    typename KernelImpl::QBMMTiling qbmmParams{
        tilingData->qbmmTilingData.baseM,
        tilingData->qbmmTilingData.baseN,
        tilingData->qbmmTilingData.baseK,
        tilingData->qbmmTilingData.dbL0c};
    typename KernelImpl::SyncParams syncParams{SyncMode};
    Params params{problemShape, mmadParams, l1Params, schedulerParams, qbmmParams, syncParams};

    KernelImpl quantMatmulKernelImpl;
    quantMatmulKernelImpl(params);
}

template<TemplateMC2TypeClass>
__aicore__ inline void allGatherMatmulImpl<TemplateMC2TypeFunc>::AllGatherProcess()
{
    allGatherMte_.InitBuffer(tpipe_);
    allGatherMte_.CopyLocalData(inputXAddr_, inputXScaleAddr_);
    allGatherMte_.SetRemoteFlag();
    allGatherMte_.WaitRemoteFlag();
    allGatherMte_.ExecuteAllGather(allGatherDataOutAddr_, allGatherScalesOutAddr_);
}

template<TemplateMC2TypeClass>
__aicore__ inline void allGatherMatmulImpl<TemplateMC2TypeFunc>::Process()
{
    if ASCEND_IS_AIV {
        AllGatherProcess();
    }
    if ASCEND_IS_AIC {
        MatmulProcess();
    }
}

} // allGatherMatmulImpl

#endif  // ALL_GATHER_MATMUL_IMPL_H
