/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software and you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file quant_matmul_reduce_scatter_impl.h
 * \brief QuantMatmul + ReduceScatter fusion kernel (1:2 CV mixed-core mode).
 *
 *        Init() — bind GM addresses and init all member variables.
 *        Process() — dispatch AIC to MatmulProcess, AIV to ReduceScatterProcess.
 */

#pragma once

#include "../utils/constant.h"
#include "adv_api/hccl/hccl.h"
#include "inc/kernel/moe_distribute_base.h"

#include "include/tensor_api/tensor.h"
#include "block/block_mmad.h"
#include "block/block_scheduler_policy.h"
#include "quant_matmul_mx_kernel_swat.h"
#include "../../../quant_matmul_reduce_scatter_tiling_data.h"


namespace quantMatmulReduceScatterImpl {

using namespace AscendC;

constexpr static uint64_t WIN_DATA_INNER_OFFSET = 30UL * 1024UL * 1024UL;
constexpr static uint64_t SINGLE_ZERONE_DATA_ZONE_SIZE = 100UL * 1024UL * 1024UL;
constexpr static uint64_t UB_ALIGN_BYTES = 32UL;
constexpr static int64_t AIV_UB_TILE_M = 32;
constexpr uint32_t FLOAT_UB_ALIGN_NUM = 8U;
constexpr uint32_t STATE_ALIGN_BYTES = 32U;
constexpr uint32_t BUFFER_NUM = 1U;
constexpr int32_t BATCH_SIZE = 1U;
constexpr int32_t SCALE_C0 = 2;

template <AscendC::HardEvent event>
__aicore__ inline void SyncFunc()
{
    int32_t eventID = static_cast<int32_t>(GetTPipePtr()->FetchEventID(event));
    AscendC::SetFlag<event>(eventID);
    AscendC::WaitFlag<event>(eventID);
}

// #define template                                                                                                       \
// <typename X1Type, typename X2Type, typename ScaleType, bool isX2Nz> template <class ProblemShape, class BlockMmad, class BlockScheduler>
// #define X1Type , X2Type, ScaleType, isX2Nz ProblemShape, BlockMmad, BlockScheduler

template <typename X1Type, typename X2Type, typename ScaleType, bool isX2Nz>
class quantMatmulReduceScatterImpl {
public:
    __aicore__ inline quantMatmulReduceScatterImpl(){};
    __aicore__ inline ~quantMatmulReduceScatterImpl(){};

    using TypeA = X1Type;
    using TypeB = X2Type;
    using TypeC = bfloat16_t;
    // int32_t SCALE_C0 = 2;
    // int32_t C0_SIZE = AscendC::AuxGetC0Size<TypeA>();

    using LayoutA = typename AscendC::Te::FrameLayoutFormat<AscendC::Te::NDExtLayoutPtn>;
    using LayoutB =
        std::conditional_t<!isX2Nz, typename AscendC::Te::FrameLayoutFormat<AscendC::Te::DNExtLayoutPtn>,
                           typename AscendC::Te::FrameLayoutFormat<AscendC::Te::NZLayoutPtn,
                                                                   AscendC::Std::Int<AscendC::AuxGetC0Size<TypeA>()>>>;
    using LayoutC = typename AscendC::Te::FrameLayoutFormat<AscendC::Te::NDExtLayoutPtn>;
    using LayoutScaleA =
        typename AscendC::Te::FrameLayoutFormat<AscendC::Te::ScaleANDLayoutPtn, AscendC::Std::Int<SCALE_C0>>;
    using LayoutScaleB = std::conditional_t<
        !isX2Nz, typename AscendC::Te::FrameLayoutFormat<AscendC::Te::ScaleBDNLayoutPtn, AscendC::Std::Int<SCALE_C0>>,
        typename AscendC::Te::FrameLayoutFormat<AscendC::Te::ScaleBNDLayoutPtn, AscendC::Std::Int<SCALE_C0>>>;

    using BlockScheduler = QuantMatmulMxSwatScheduler<NO_FULL_LOAD_MODE>;
    using DispatchPolicy = QuantMatmulMxMultiBlockWithSwat<NO_FULL_LOAD_MODE, 2UL>;
    using ProblemShape = MatmulShape;
    using TypeATuple = AscendC::Std::tuple<TypeA, ScaleType>;
    using TypeBTuple = AscendC::Std::tuple<TypeB, ScaleType>;
    using LayoutATuple = AscendC::Std::tuple<LayoutA, LayoutScaleA>;
    using LayoutBTuple = AscendC::Std::tuple<LayoutB, LayoutScaleB>;
    using BlockMmad =
        Block::BlockMmad<DispatchPolicy, TypeATuple, LayoutATuple, TypeBTuple, LayoutBTuple, TypeC, LayoutC>;
    using KernelImpl = Kernel::QuantMatmulMxKernelSwat<ProblemShape, BlockMmad, BlockScheduler>;
    using Params = typename KernelImpl::Params;

    // static constexpr bool weightNz = BlockMmad::weightNz;
    // static constexpr bool transA = BlockMmad::transA;
    // static constexpr bool transB = BlockMmad::transB;
    // using AType = typename BlockMmad::AType;
    // using BType = typename BlockMmad::BType;
    // using ScaleAType = typename BlockMmad::ScaleAType;
    // using ScaleBType = typename BlockMmad::ScaleBType;
    // using CType = typename BlockMmad::CType;

    // The kernel layer does not own a scheduling policy itself; it selects the
    // concrete scheduler from the block pipeline traits.
    using BlockSchedulerOp =
        typename Block::BlockSchedulerSelector<ProblemShape, BlockScheduler, BlockMmad::transA, BlockMmad::transB,
                                               typename BlockMmad::AType>::SchedulerOp;

    using BlockMmadParams = typename BlockMmad::Params;
    using L1Params = typename BlockMmad::L1Params;

    using TupleShape = AscendC::Shape<int64_t, int64_t, int64_t>;
    using BlockShape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockCoord = AscendC::Coord<int64_t, int64_t, int64_t, int64_t>;
    using BlockSchedulerParams = typename BlockSchedulerOp::Params;

    using MakeLayoutA = typename BlockMmad::LayoutA;
    using MakeLayoutB = typename BlockMmad::LayoutB;
    using MakeLayoutScaleA = typename BlockMmad::LayoutScaleA;
    using MakeLayoutScaleB = typename BlockMmad::LayoutScaleB;

    // 这两个params的参数不再做成struct，直接改成private在当前方法中使用
    // struct CommParams {
    //     int64_t tpSizeM;
    //     int32_t tpWorldSize;
    // };

    // struct OutputParams {
    //     GM_ADDR yOut = nullptr;
    // };

    // struct Params {
    //     ProblemShape problemShape;
    //     typename BlockMmad::Params mmadParams;
    //     typename BlockMmad::L1Params l1Params;
    //     BlockSchedulerOp::Params schParams;
    //     typename BlockMmad::QBMMTiling qbmmParams;
    //     CommParams commParams;
    //     OutputParams outputParams;
    // };

    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR x1Scale, GM_ADDR x2Scale, GM_ADDR y,
                                GM_ADDR workspaceGm, GM_ADDR tilingGM);
    __aicore__ inline void Process();

private:
    __aicore__ inline void InitMatmulParams();
    __aicore__ inline void InitReduceScatterBuffer();
    __aicore__ inline void MatmulProcess();
    __aicore__ inline void ReduceScatterProcess();
    __aicore__ inline void CommProcess(int64_t mPos, int64_t nPos, const BlockShape &singleShape);
    __aicore__ inline void WriteStatusToWin();
    __aicore__ inline void ReadStatus();
    __aicore__ inline void ReadRemoteDataAdd();
    __aicore__ inline GM_ADDR GetWindAddrByRankId(const int32_t rankId);
    __aicore__ inline GM_ADDR GetWindStateAddrByRankId(const int32_t rankId);
    __aicore__ inline void SplitToCore(const uint32_t curSendCnt, const uint32_t curUseAivNum, const uint32_t coreId,
                                       uint32_t &startId, uint32_t &endId, uint32_t &sendNum);

private:
    // BlockMmad mmadOp_;
    // __gm__ X1Type *aGmAddr_{nullptr};
    // __gm__ X2Type *bGmAddr_{nullptr};
    // __gm__ bfloat16_t *cGmAddr_{nullptr};
    // __gm__ ScaleType *scaleAGmAddr_{nullptr};
    // __gm__ ScaleType *scaleBGmAddr_{nullptr};
    // __gm__ bfloat16_t *yGmAddr_{nullptr};

    __gm__ Mc2Kernel::HcclOpParam *winContext_{nullptr};
    __gm__ QuantMatmulReduceScatterTilingData *tilingData;

    TPipe *tpipe_{nullptr};
    uint32_t rankId_{0};
    uint32_t coreVid_{0};

    int64_t tpSizeM_{0};
    uint32_t tpWorldSize_{0};

    uint64_t m_{0};
    uint64_t k_{0};
    uint64_t n_{0};
    GM_ADDR x1GM_;
    GM_ADDR x2GM_;
    GM_ADDR x1ScaleGM_;
    GM_ADDR x2ScaleGM_;
    GM_ADDR cGM_;
    GM_ADDR yGM_;

    Params params_;
    // BlockSchedulerOp bs_;

    AscendC::TQueBind<AscendC::TPosition::VECIN, AscendC::TPosition::VECOUT, BUFFER_NUM> inQueue_;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> tokenQueue_;
    AscendC::TBuf<> writeStateBuf_;
    AscendC::TBuf<> readStateBuf_;
    AscendC::TBuf<> resetStateBuf_;
    AscendC::TBuf<> sumFp32Buf_;
    AscendC::TBuf<> tokenFp32Buf_;
    AscendC::TBuf<> tokenBuf_;

    AscendC::GlobalTensor<bfloat16_t> yOutGM_;
    AscendC::LocalTensor<bfloat16_t> tokenTensor_;
};

template <typename X1Type, typename X2Type, typename ScaleType, bool isX2Nz>
__aicore__ inline void quantMatmulReduceScatterImpl<X1Type, X2Type, ScaleType, isX2Nz>::InitMatmulParams()
{
    ProblemShape problemShape{tilingData->qbmmTilingdata.m, tilingData->qbmmTilingdata.n, tilingData->qbmmTilingdata.k,
                              BATCH_SIZE};
    typename BlockMmad::Params mmadParams{x1GM_, x2GM_, cGM_, x1ScaleGM_, x2ScaleGM_};
    typename KernelImpl::L1Params l1Params{static_cast<uint64_t>(tilingData->qbmmTilingdata.stepK) *
                                      tilingData->qbmmTilingdata.baseK,
                                  tilingData->qbmmTilingdata.scaleKL1};
    typename KernelImpl::BlockSchedulerParams schedulerParams{tilingData->qbmmTilingdata.baseM,
                                                     tilingData->qbmmTilingdata.baseN,
                                                     tilingData->qbmmTilingdata.mTailTile,
                                                     tilingData->qbmmTilingdata.nTailTile,
                                                     tilingData->qbmmTilingdata.mBaseTailSplitCnt,
                                                     tilingData->qbmmTilingdata.nBaseTailSplitCnt,
                                                     tilingData->qbmmTilingdata.mTailMain,
                                                     tilingData->qbmmTilingdata.nTailMain};
    typename KernelImpl::QBMMTiling qbmmParams{tilingData->qbmmTilingdata.baseM, tilingData->qbmmTilingdata.baseN,
                                      tilingData->qbmmTilingdata.baseK, tilingData->qbmmTilingdata.dbL0c};
    // CommParams commParams{problemShape.m / tilingData->quantMatmulReduceScatterTilingInfo.rankSize,
    //                       static_cast<int32_t>(tilingData.quantMatmulReduceScatterTilingInfo.rankSize)};
    // OutputParams outputParams{y};
    params_ = {problemShape, mmadParams, l1Params, schedulerParams, qbmmParams};
}

template <typename X1Type, typename X2Type, typename ScaleType, bool isX2Nz>
__aicore__ inline void quantMatmulReduceScatterImpl<X1Type, X2Type, ScaleType, isX2Nz>::InitReduceScatterBuffer()
{
    tpipe_->Reset();
    tpipe_->InitBuffer(tokenQueue_, BUFFER_NUM, params_.problemShape.n * sizeof(float));
    tpipe_->InitBuffer(sumFp32Buf_, params_.problemShape.n * sizeof(float));
    tpipe_->InitBuffer(tokenFp32Buf_, params_.problemShape.n * sizeof(float));
    tpipe_->InitBuffer(tokenBuf_, params_.problemShape.n * sizeof(bfloat16_t));
}

template <typename X1Type, typename X2Type, typename ScaleType, bool isX2Nz>
__aicore__ inline void quantMatmulReduceScatterImpl<X1Type, X2Type, ScaleType, isX2Nz>::Init(
    GM_ADDR x1, GM_ADDR x2, GM_ADDR x1Scale, GM_ADDR x2Scale, GM_ADDR y, GM_ADDR workspaceGm, GM_ADDR tilingGM)
{
    // params_ = params;

    // aGmAddr_ = reinterpret_cast<__gm__ AType *>(params.mmadParams.aGmAddr);
    // bGmAddr_ = reinterpret_cast<__gm__ BType *>(params.mmadParams.bGmAddr);
    // cGmAddr_ = reinterpret_cast<__gm__ CType *>(params.mmadParams.cGmAddr);
    // scaleAGmAddr_ = reinterpret_cast<__gm__ fp8_e8m0_t *>(params.mmadParams.scaleAGmAddr);
    // scaleBGmAddr_ = reinterpret_cast<__gm__ fp8_e8m0_t *>(params.mmadParams.scaleBGmAddr);
    // yGmAddr_ = reinterpret_cast<__gm__ bfloat16_t *>(params.outputParams.yOut);


    x1GM_ = x1;
    x2GM_ = x2;
    x1ScaleGM_ = x1Scale;
    x2ScaleGM_ = x2Scale;
    cGM_ = !isX2Nz ? y : workspaceGm;
    yGM_ = y;

    tilingData = (__gm__ QuantMatmulReduceScatterTilingData *)tilingGM;
    m_ = tilingData->qbmmTilingdata.m;
    k_ = tilingData->qbmmTilingdata.k;
    n_ = tilingData->qbmmTilingdata.n;

    tpSizeM_ = m_ / tilingData->quantMatmulReduceScatterTilingInfo.rankSize;
    tpWorldSize_ = static_cast<int32_t>(tilingData->quantMatmulReduceScatterTilingInfo.rankSize);

    // bs_ = BlockSchedulerOp(params.problemShape, params.schParams);

    // if ASCEND_IS_AIC {
    //     TupleShape problemShape_ = {params.problemShape.m, params.problemShape.n, params.problemShape.k};
    //     BlockShape l0TileShape{params.qbmmParams.baseM, params.qbmmParams.baseN, params.qbmmParams.baseK, 0};
    //     bool enableL0cPingPong = (params.qbmmParams.dbL0C > 1);
    //     mmadOp_.Init(problemShape_, l0TileShape, params.l1Params, enableL0cPingPong);
    //     return;
    // }

    if ASCEND_IS_AIV {
        TPipe pipe;
        tpipe_ = &pipe;
        if (tpipe_ != nullptr) {
            uint32_t paddedRowBytes =
                ::CeilDiv(static_cast<uint64_t>(tilingData->qbmmTilingdata.baseN) * sizeof(bfloat16_t),
                          UB_ALIGN_BYTES) *
                UB_ALIGN_BYTES;
            tpipe_->InitBuffer(inQueue_, BUFFER_NUM, AIV_UB_TILE_M * paddedRowBytes);
            tpipe_->InitBuffer(writeStateBuf_, STATE_ALIGN_BYTES);
            tpipe_->InitBuffer(readStateBuf_, STATE_ALIGN_BYTES);
            tpipe_->InitBuffer(resetStateBuf_, STATE_ALIGN_BYTES);
        }
        winContext_ = (__gm__ Mc2Kernel::HcclOpParam *)AscendC::GetHcclContext<HCCL_GROUP_ID_0>();
        rankId_ = Mc2Kernel::GetRankId(winContext_);
        coreVid_ = AscendC::GetBlockIdx();

        yOutGM_.SetGlobalBuffer((__gm__ bfloat16_t*)yGM_);
    }

    // matmul init in tensor api mode
    InitMatmulParams();
}

template <typename X1Type, typename X2Type, typename ScaleType, bool isX2Nz>
__aicore__ inline void quantMatmulReduceScatterImpl<X1Type, X2Type, ScaleType, isX2Nz>::Process()
{
    if ASCEND_IS_AIC {
        MatmulProcess();
    }
    if ASCEND_IS_AIV {
        ReduceScatterProcess();
    }
}

template <typename X1Type, typename X2Type, typename ScaleType, bool isX2Nz>
__aicore__ inline void quantMatmulReduceScatterImpl<X1Type, X2Type, ScaleType, isX2Nz>::MatmulProcess()
{
    // using TypeA = X1Type;
    // using TypeB = X2Type;
    // using TypeC = bfloat16_t;
    // constexpr int32_t SCALE_C0 = 2;
    // constexpr int32_t C0_SIZE = AscendC::AuxGetC0Size<TypeA>();
    // using LayoutA = typename AscendC::Te::FrameLayoutFormat<AscendC::Te::NDExtLayoutPtn>;
    // using LayoutB = isX2Nz ?
    //                     typename AscendC::Te::FrameLayoutFormat<AscendC::Te::DNExtLayoutPtn> :
    //                     typename AscendC::Te::FrameLayoutFormat<AscendC::Te::NZLayoutPtn,
    //                     AscendC::Std::Int<C0_SIZE>>;
    // using LayoutC = typename AscendC::Te::FrameLayoutFormat<AscendC::Te::NDExtLayoutPtn>;
    // using LayoutScaleA =
    //     typename AscendC::Te::FrameLayoutFormat<AscendC::Te::ScaleANDLayoutPtn, AscendC::Std::Int<SCALE_C0>>;
    // using LayoutScaleB =
    //     isX2Nz ? typename AscendC::Te::FrameLayoutFormat<AscendC::Te::ScaleBDNLayoutPtn, AscendC::Std::Int<SCALE_C0>>
    //     :
    //              typename AscendC::Te::FrameLayoutFormat<AscendC::Te::ScaleBNDLayoutPtn,
    //              AscendC::Std::Int<SCALE_C0>>;

    // using BlockScheduler = QuantMatmulMxSwatScheduler<NO_FULL_LOAD_MODE>;
    // using DispatchPolicy = QuantMatmulMxMultiBlockWithSwat<NO_FULL_LOAD_MODE, 2UL>;
    // using ProblemShape = MatmulShape;
    // using TypeATuple = AscendC::Std::tuple<TypeA, ScaleType>;
    // using TypeBTuple = AscendC::Std::tuple<TypeB, ScaleType>;
    // using LayoutATuple = AscendC::Std::tuple<LayoutA, LayoutScaleA>;
    // using LayoutBTuple = AscendC::Std::tuple<LayoutB, LayoutScaleB>;
    // using BlockMmad =
    //     Block::BlockMmad<DispatchPolicy, TypeATuple, LayoutATuple, TypeBTuple, LayoutBTuple, TypeC, LayoutC>;
    // using KernelImpl = Kernel::QuantMatmulMxKernelSwat<ProblemShape, BlockMmad, BlockScheduler>;
    // using Params = typename KernelImpl::Params;

    // using CommParams = typename KernelImpl::CommParams;
    // using OutputParams = typename KernelImpl::OutputParams;

    // ProblemShape problemShape{tilingData->qbmmTilingdata.m, tilingData->qbmmTilingdata.n,
    // tilingData->qbmmTilingdata.k,
    //                           BATCH_SIZE};
    // BlockMmad::Params mmadParams{x1GM_, x2GM_, cGM_, x1ScaleGM_, x2ScaleGM_};
    // KernelImpl::L1Params l1Params{static_cast<uint64_t>(tilingData->qbmmTilingdata.stepK) *
    //                                   tilingData->qbmmTilingdata.baseK,
    //                               tilingData->qbmmTilingdata.scaleKL1};
    // KernelImpl::BlockSchedulerParams schedulerParams{tilingData->qbmmTilingdata.baseM,
    //                                                  tilingData->qbmmTilingdata.baseN,
    //                                                  tilingData->qbmmTilingdata.mTailTile,
    //                                                  tilingData->qbmmTilingdata.nTailTile,
    //                                                  tilingData->qbmmTilingdata.mBaseTailSplitCnt,
    //                                                  tilingData->qbmmTilingdata.nBaseTailSplitCnt,
    //                                                  tilingData->qbmmTilingdata.mTailMain,
    //                                                  tilingData->qbmmTilingdata.nTailMain};
    // KernelImpl::QBMMTiling qbmmParams{tilingData->qbmmTilingdata.baseM, tilingData->qbmmTilingdata.baseN,
    //                                   tilingData->qbmmTilingdata.baseK, tilingData->qbmmTilingdata.dbL0c};
    // // CommParams commParams{problemShape.m / tilingData->quantMatmulReduceScatterTilingInfo.rankSize,
    // //                       static_cast<int32_t>(tilingData.quantMatmulReduceScatterTilingInfo.rankSize)};
    // // OutputParams outputParams{y};
    // Params params{problemShape, mmadParams, l1Params, schedulerParams, qbmmParams};
    KernelImpl kernel;
    kernel(params_);
}

template <typename X1Type, typename X2Type, typename ScaleType, bool isX2Nz>
__aicore__ inline void quantMatmulReduceScatterImpl<X1Type, X2Type, ScaleType, isX2Nz>::ReduceScatterProcess()
{
    BlockSchedulerOp bs(params_.problemShape, params_.schParams);

    BlockCoord blockIdx;
    while (bs.template GetTileIdx<BlockMmad::weightNz>(blockIdx)) {
        int64_t mPos = AscendC::Te::Get<MNK_M>(blockIdx);
        int64_t nPos = AscendC::Te::Get<MNK_N>(blockIdx);
        BlockShape singleShape = bs.template GetBlockShape<BlockMmad::weightNz>(blockIdx);
        if (AscendC::Te::Get<MNK_M>(singleShape) <= 0 || AscendC::Te::Get<MNK_N>(singleShape) <= 0) {
            break;
        }
        CrossCoreWaitFlag(9);
        CommProcess(mPos, nPos, singleShape);
    }

    SyncAll<true>();
    WriteStatusToWin();
    ReadStatus();
    SyncAll<true>();

    // tpipe_->Reset();
    // tpipe_->InitBuffer(tokenQueue_, BUFFER_NUM, params_.problemShape.n * sizeof(float));
    // tpipe_->InitBuffer(sumFp32Buf_, params_.problemShape.n * sizeof(float));
    // tpipe_->InitBuffer(tokenFp32Buf_, params_.problemShape.n * sizeof(float));
    // tpipe_->InitBuffer(tokenBuf_, params_.problemShape.n * sizeof(bfloat16_t));

    InitReduceScatterBuffer();

    ReadRemoteDataAdd();
}

template <typename X1Type, typename X2Type, typename ScaleType, bool isX2Nz>
__aicore__ inline GM_ADDR
quantMatmulReduceScatterImpl<X1Type, X2Type, ScaleType, isX2Nz>::GetWindAddrByRankId(const int32_t rankId)
{
    return Mc2Kernel::GetBaseWindAddrByRankId(winContext_, rankId, rankId_);
}

template <typename X1Type, typename X2Type, typename ScaleType, bool isX2Nz>
__aicore__ inline GM_ADDR
quantMatmulReduceScatterImpl<X1Type, X2Type, ScaleType, isX2Nz>::GetWindStateAddrByRankId(const int32_t rankId)
{
    return Mc2Kernel::GetBaseWindStateAddrByRankId(winContext_, rankId, rankId_);
}

template <typename X1Type, typename X2Type, typename ScaleType, bool isX2Nz>
__aicore__ inline void quantMatmulReduceScatterImpl<X1Type, X2Type, ScaleType, isX2Nz>::SplitToCore(
    const uint32_t curSendCnt, const uint32_t curUseCoreNum, const uint32_t coreId, uint32_t &startId, uint32_t &endId,
    uint32_t &sendNum)
{
    sendNum = curSendCnt / curUseCoreNum;
    uint32_t remainderNum = curSendCnt % curUseCoreNum;
    startId = sendNum * coreId;
    if (coreId < remainderNum) {
        sendNum += 1;
        startId += coreId;
    } else {
        startId += remainderNum;
    }
    endId = startId + sendNum;
}

template <typename X1Type, typename X2Type, typename ScaleType, bool isX2Nz>
__aicore__ inline void
quantMatmulReduceScatterImpl<X1Type, X2Type, ScaleType, isX2Nz>::CommProcess(int64_t mPos, int64_t nPos,
                                                                             const BlockShape &singleShape)
{
    int64_t singleM = AscendC::Te::Get<MNK_M>(singleShape);
    int64_t singleN = AscendC::Te::Get<MNK_N>(singleShape);
    if (singleM <= 0 || singleN <= 0) {
        return;
    }

    int64_t tpSizeM = tpSizeM_;
    int64_t aivIdx = coreVid_ % 2;
    int64_t subTileCnt = ::CeilDiv(singleM, AIV_UB_TILE_M);

    __gm__ TypeC *workspaceAddr = reinterpret_cast<__gm__ TypeC *>(params_.mmadParams.cGmAddr);
    AscendC::GlobalTensor<bfloat16_t> workspaceGm;
    workspaceGm.SetGlobalBuffer(workspaceAddr);

    uint32_t blkLenBytes = static_cast<uint32_t>(singleN * sizeof(bfloat16_t));
    uint32_t rowPaddedBytes = ::CeilDiv(blkLenBytes, UB_ALIGN_BYTES) * UB_ALIGN_BYTES;
    uint32_t rowInVECINElems = rowPaddedBytes / sizeof(bfloat16_t);

    for (int64_t st = aivIdx; st < subTileCnt; st += 2) {
        int64_t subTileM = Min(AIV_UB_TILE_M, singleM - st * AIV_UB_TILE_M);
        if (subTileM <= 0) {
            continue;
        }
        int64_t globalBaseM = mPos + st * AIV_UB_TILE_M;

        DataCopyPadExtParams<bfloat16_t> padParams{false, 0, 0, 0};
        AscendC::LocalTensor<bfloat16_t> batchLocal = inQueue_.AllocTensor<bfloat16_t>();

        DataCopyExtParams readParams{static_cast<uint16_t>(subTileM), blkLenBytes,
                                     static_cast<uint32_t>((n_ - singleN) * sizeof(bfloat16_t)), 0, 0};
        DataCopyPad(batchLocal, workspaceGm[globalBaseM * n_ + nPos], readParams, padParams);
        inQueue_.EnQue(batchLocal);
        batchLocal = inQueue_.DeQue<bfloat16_t>();

        int64_t processed = 0;
        while (processed < subTileM) {
            int64_t globalRow = globalBaseM + processed;
            int32_t dstRankId = static_cast<int32_t>(globalRow / tpSizeM);
            if (dstRankId >= tpWorldSize_) {
                break;
            }
            int64_t tpBoundary = (dstRankId + 1) * tpSizeM;
            int64_t rowsInBatch = Min(subTileM - processed, tpBoundary - globalRow);
            if (rowsInBatch <= 0) {
                break;
            }

            GM_ADDR remoteWinAddr = GetWindAddrByRankId(dstRankId);

            AscendC::GlobalTensor<bfloat16_t> remoteGm;
            remoteGm.SetGlobalBuffer((__gm__ bfloat16_t *)remoteWinAddr);
            uint64_t dstBaseOffset = static_cast<uint64_t>(rankId_) * tpSizeM * n_ + (globalRow % tpSizeM) * n_ + nPos;

            DataCopyExtParams writeParams{static_cast<uint16_t>(rowsInBatch), blkLenBytes, 0,
                                          static_cast<uint32_t>((n_ - singleN) * sizeof(bfloat16_t)), 0};
            DataCopyPad(remoteGm[dstBaseOffset], batchLocal[processed * rowInVECINElems], writeParams);

            processed += rowsInBatch;
        }

        inQueue_.FreeTensor(batchLocal);
    }
}

template <typename X1Type, typename X2Type, typename ScaleType, bool isX2Nz>
__aicore__ inline void quantMatmulReduceScatterImpl<X1Type, X2Type, ScaleType, isX2Nz>::ReadRemoteDataAdd()
{
    uint32_t startRowId = 0;
    uint32_t endRowId = 0;
    uint32_t rowNum = 0;
    uint32_t tokenIndex = 0;
    SplitToCore(m_ / tpWorldSize_, GetBlockNum() * 2, coreVid_, startRowId, endRowId, rowNum);

    AscendC::LocalTensor<float> sumFp32Tensor = sumFp32Buf_.Get<float>();
    AscendC::LocalTensor<float> tokenFp32Tensor = tokenFp32Buf_.Get<float>();
    AscendC::GlobalTensor<bfloat16_t> localWinTensor;
    localWinTensor.SetGlobalBuffer((__gm__ bfloat16_t *)GetWindAddrByRankId(rankId_));

    DataCopyExtParams expandXCopyParams{1U, static_cast<uint32_t>(n_ * sizeof(bfloat16_t)), 0U, 0U, 0U};

    for (uint32_t tileIdx = 0; tileIdx < rowNum; ++tileIdx) {
        Duplicate<float>(sumFp32Tensor, (float)0.0, n_);
        tokenIndex = startRowId + tileIdx;
        uint32_t curVidTileOffset = tokenIndex * n_;
        for (int tpIndex = 0; tpIndex < tpWorldSize_; ++tpIndex) {
            uint32_t curOffset = curVidTileOffset + tpIndex * (m_ / tpWorldSize_ * n_);
            tokenTensor_ = tokenQueue_.AllocTensor<bfloat16_t>();
            DataCopy(tokenTensor_, localWinTensor[curOffset], n_);
            tokenQueue_.EnQue(tokenTensor_);
            tokenTensor_ = tokenQueue_.DeQue<bfloat16_t>();
            SyncFunc<AscendC::HardEvent::MTE2_V>();
            Cast(tokenFp32Tensor, tokenTensor_, AscendC::RoundMode::CAST_NONE, n_);
            AscendC::PipeBarrier<PIPE_V>();
            Add(sumFp32Tensor, sumFp32Tensor, tokenFp32Tensor, n_);
            tokenQueue_.FreeTensor<bfloat16_t>(tokenTensor_);
        }

        AscendC::LocalTensor<bfloat16_t> sumBufLocal = tokenBuf_.Get<bfloat16_t>();
        AscendC::PipeBarrier<PIPE_V>();
        Cast(sumBufLocal, sumFp32Tensor, AscendC::RoundMode::CAST_RINT, n_);
        SyncFunc<AscendC::HardEvent::V_MTE3>();
        DataCopyPad(yOutGM_[tokenIndex * n_], sumBufLocal, expandXCopyParams);
    }
}

template <typename X1Type, typename X2Type, typename ScaleType, bool isX2Nz>
__aicore__ inline void quantMatmulReduceScatterImpl<X1Type, X2Type, ScaleType, isX2Nz>::WriteStatusToWin()
{
    if (coreVid_ >= tpWorldSize_) {
        return;
    }
    uint32_t curOffset = rankId_ * FLOAT_UB_ALIGN_NUM;
    AscendC::LocalTensor<float> statusTensor = writeStateBuf_.Get<float>();
    statusTensor(0) = (float)1;
    uint32_t curDstId = coreVid_;
    GM_ADDR remoteWinStateGM = GetWindStateAddrByRankId(curDstId);
    AscendC::GlobalTensor<float> stateGMTensor;
    stateGMTensor.SetGlobalBuffer((__gm__ float *)remoteWinStateGM);
    SyncFunc<AscendC::HardEvent::S_MTE3>();
    DataCopy(stateGMTensor[curOffset], statusTensor, FLOAT_UB_ALIGN_NUM);
}

template <typename X1Type, typename X2Type, typename ScaleType, bool isX2Nz>
__aicore__ inline void quantMatmulReduceScatterImpl<X1Type, X2Type, ScaleType, isX2Nz>::ReadStatus()
{
    if (coreVid_ >= tpWorldSize_) {
        return;
    }
    GM_ADDR stateGM = GetWindStateAddrByRankId(rankId_);
    AscendC::GlobalTensor<float> selfStateWinTensor;
    selfStateWinTensor.SetGlobalBuffer((__gm__ float *)(stateGM));
    uint32_t offset = coreVid_ * FLOAT_UB_ALIGN_NUM;
    AscendC::LocalTensor<float> statusTensor = readStateBuf_.Get<float>();
    AscendC::LocalTensor<float> resetStatusTensor = resetStateBuf_.Get<float>();
    Duplicate<float>(resetStatusTensor, 0.0, FLOAT_UB_ALIGN_NUM);
    float flag = -1;
    uint32_t statusCnt = FLOAT_UB_ALIGN_NUM;
    float minTarget = (float)0.5;
    float maxTarget = (float)1.5;
    while ((flag < minTarget) || (flag > maxTarget)) {
        SyncFunc<AscendC::HardEvent::S_MTE2>();
        DataCopy(statusTensor, selfStateWinTensor[offset], statusCnt);
        SyncFunc<AscendC::HardEvent::MTE2_S>();
        flag = statusTensor(0);
    }
    SyncFunc<AscendC::HardEvent::V_MTE3>();
    SyncFunc<AscendC::HardEvent::MTE2_MTE3>();
    DataCopy(selfStateWinTensor[offset], resetStatusTensor, statusCnt);
}

} // namespace quantMatmulReduceScatterImpl