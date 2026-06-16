/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software and you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR OR FITNESS FOR A PARTICULAR PURPOSE.
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

#include "adv_api/hccl/hccl.h"
#include "inc/kernel/moe_distribute_base.h"
#include "include/tensor_api/tensor.h"
#include "blaze/gemm/utils/common_utils.h"
#include "blaze/gemm/utils/layout_utils.h"
#include "blaze/gemm/policy/dispatch_policy.h"
#include "blaze/gemm/block/block_mmad.h"
#include "blaze/gemm/block/block_mmad_qbmm_mx.h"
#include "blaze/gemm/block/block_scheduler_qbmm.h"
#include "blaze/epilogue/block/block_epilogue_empty.h"
#include "blaze/gemm/kernel/kernel_universal.h"
#include "quant_matmul_mx_kernel_swat_qmmrs.h" // 定制kernel 层
#include "quant_matmul_reduce_scatter_tiling_data.h"

namespace quantMatmulReduceScatterImpl {

using namespace AscendC;
using namespace Blaze::Gemm;

constexpr static uint64_t WIN_DATA_INNER_OFFSET = 30UL * 1024UL * 1024UL;
constexpr static uint64_t SINGLE_ZERONE_DATA_ZONE_SIZE = 100UL * 1024UL * 1024UL;
constexpr static uint64_t UB_ALIGN_BYTES = 32UL;
constexpr static int64_t AIV_UB_TILE_M = 32;
constexpr uint32_t FLOAT_UB_ALIGN_NUM = 8U;
constexpr uint32_t STATE_ALIGN_BYTES = 32U;
constexpr uint32_t BUFFER_NUM = 1U;
constexpr int64_t BATCH_SIZE = 1U;
constexpr int32_t L1_BUF_NUM = 2;
constexpr uint16_t CROSS_CORE_FLAG_ID_NINE = 9U;

template <AscendC::HardEvent event>
__aicore__ inline void SyncFunc()
{
    int32_t eventID = static_cast<int32_t>(GetTPipePtr()->FetchEventID(event));
    AscendC::SetFlag<event>(eventID);
    AscendC::WaitFlag<event>(eventID);
}

template <typename X1Type, typename X2Type, typename ScaleType, bool isX2Nz>
class quantMatmulReduceScatterImpl {
public:
    __aicore__ inline quantMatmulReduceScatterImpl(){};
    __aicore__ inline ~quantMatmulReduceScatterImpl(){};

    using TypeA = X1Type;
    using TypeB = X2Type;
    using TypeC = bfloat16_t;
    using BiasType = float;

    using LayoutA = AscendC::Te::NDExtLayoutPtn;
    using LayoutB = std::conditional_t<!isX2Nz, AscendC::Te::DNExtLayoutPtn, AscendC::Te::NZLayoutPtn>;
    using LayoutC = AscendC::Te::NDExtLayoutPtn;
    using LayoutScaleA = AscendC::Te::ScaleANDLayoutPtn;
    using LayoutScaleB = std::conditional_t<!isX2Nz, AscendC::Te::ScaleBDNLayoutPtn, AscendC::Te::ScaleBNDLayoutPtn>;
    using DispatchPolicy = Blaze::Gemm::MatmulWithScaleMx<0, 0>;
    using BlockMmad = Blaze::Gemm::Block::BlockMmad<DispatchPolicy, TypeA, LayoutA, TypeB, LayoutB, TypeC, LayoutC,
                                                    BiasType, LayoutC>;
    using BlockScheduler =
        Blaze::Gemm::Block::BlockSchedulerQuantBatchMatmulV3<AscendC::Te::Shape<int64_t, int64_t, int64_t, int64_t>, 0,
                                                             LayoutA, LayoutB, TypeA>;
    using BlockEpilogue = Blaze::Gemm::Block::BlockEpilogueEmpty;
    using ProblemShape = AscendC::Te::Shape<int64_t, int64_t, int64_t, int64_t>;
    using KernelImpl = Blaze::Gemm::Kernel::GemmUniversal<
        ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler,
        AscendC::Std::enable_if_t<AscendC::Std::is_same_v<Blaze::Gemm::KernelMmadWithScaleMx,
                                                          typename BlockMmad::DispatchPolicy::ScheduleType>>>;
    using Params = typename KernelImpl::Params;

    using BlockMmadParams = typename BlockMmad::Params;
    using L1Params = typename BlockMmad::L1Params;
    using BlockSchedulerParams = typename BlockScheduler::Params;

    using BlockShape = AscendC::Te::Shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockCoord = AscendC::Te::Coord<int64_t, int64_t, int64_t, int64_t>;

    using MakeLayoutUB = AscendC::Te::FrameLayoutFormat<AscendC::Te::NDExtLayoutPtn, AscendC::Te::LayoutTraitDefault<TypeC>>;

    static constexpr bool weightNz = BlockMmad::weightNz;

    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR x1Scale, GM_ADDR x2Scale, GM_ADDR y,
                                GM_ADDR workspaceGm, GM_ADDR tilingGM);
    __aicore__ inline void Process();

private:
    __aicore__ inline void InitMatmulParams();
    __aicore__ inline void InitReduceScatterBuffer();
    __aicore__ inline void MatmulProcess();
    __aicore__ inline void ReduceScatterProcess();
    __aicore__ inline void CommProcess(int64_t mPos, int64_t nPos, const BlockShape &singleShape);
    __aicore__ inline void CommProcessTensorMode(int64_t mPos, int64_t nPos, const BlockShape &singleShape);
    __aicore__ inline void WriteStatusToWin();
    __aicore__ inline void ReadStatus();
    __aicore__ inline void ReadRemoteDataAdd();
    __aicore__ inline GM_ADDR GetWindAddrByRankId(const int32_t rankId);
    __aicore__ inline GM_ADDR GetWindStateAddrByRankId(const int32_t rankId);
    __aicore__ inline void SplitToCore(const uint32_t curSendCnt, const uint32_t curUseAivNum, const uint32_t coreId,
                                       uint32_t &startId, uint32_t &endId, uint32_t &sendNum);

private:
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

    __gm__ TypeC* cGmAddr_;

    Params params_;

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
    ProblemShape problemShape{static_cast<uint64_t>(tilingData->qbmmTilingdata.m),
                              static_cast<uint64_t>(tilingData->qbmmTilingdata.n),
                              static_cast<uint64_t>(tilingData->qbmmTilingdata.k), BATCH_SIZE};
    typename BlockMmad::Params mmadParams{x1GM_, x2GM_, cGM_, nullptr, x1ScaleGM_, x2ScaleGM_};
    typename BlockMmad::L1Params l1Params{static_cast<uint64_t>(tilingData->qbmmTilingdata.stepK) *
                                              tilingData->qbmmTilingdata.baseK,
                                          tilingData->qbmmTilingdata.scaleKL1, L1_BUF_NUM};
    typename BlockScheduler::Params schedulerParams{tilingData->qbmmTilingdata.baseM,
                                                    tilingData->qbmmTilingdata.baseN,
                                                    tilingData->qbmmTilingdata.mTailTile,
                                                    tilingData->qbmmTilingdata.nTailTile,
                                                    tilingData->qbmmTilingdata.mBaseTailSplitCnt,
                                                    tilingData->qbmmTilingdata.nBaseTailSplitCnt,
                                                    tilingData->qbmmTilingdata.mTailMain,
                                                    tilingData->qbmmTilingdata.nTailMain};
    typename KernelImpl::QBMMTiling qbmmParams{
        1, 1, 1, 1,
        1, 1, 1, 1,
        1, 1, 1, 1,
        0,
        tilingData->qbmmTilingdata.baseM,
        tilingData->qbmmTilingdata.baseN,
        tilingData->qbmmTilingdata.baseK,
        0,
        tilingData->qbmmTilingdata.dbL0c};
    params_ = {problemShape, mmadParams, l1Params, schedulerParams, qbmmParams};
}

template <typename X1Type, typename X2Type, typename ScaleType, bool isX2Nz>
__aicore__ inline void quantMatmulReduceScatterImpl<X1Type, X2Type, ScaleType, isX2Nz>::InitReduceScatterBuffer()
{
    tpipe_->Reset();
    tpipe_->InitBuffer(tokenQueue_, BUFFER_NUM, AscendC::Te::Get<MNK_N>(params_.problemShape) * sizeof(float));
    tpipe_->InitBuffer(sumFp32Buf_, AscendC::Te::Get<MNK_N>(params_.problemShape) * sizeof(float));
    tpipe_->InitBuffer(tokenFp32Buf_, AscendC::Te::Get<MNK_N>(params_.problemShape) * sizeof(float));
    tpipe_->InitBuffer(tokenBuf_, AscendC::Te::Get<MNK_N>(params_.problemShape) * sizeof(bfloat16_t));
}

template <typename X1Type, typename X2Type, typename ScaleType, bool isX2Nz>
__aicore__ inline void quantMatmulReduceScatterImpl<X1Type, X2Type, ScaleType, isX2Nz>::Init(
    GM_ADDR x1, GM_ADDR x2, GM_ADDR x1Scale, GM_ADDR x2Scale, GM_ADDR y, GM_ADDR workspaceGm, GM_ADDR tilingGM)
{
    x1GM_ = x1;
    x2GM_ = x2;
    x1ScaleGM_ = x1Scale;
    x2ScaleGM_ = x2Scale;
    cGM_ = !isX2Nz ? y : workspaceGm;
    yGM_ = y;

    cGmAddr_ = reinterpret_cast<__gm__ TypeC*>(cGM_); // tensor 化通信

    tilingData = (__gm__ QuantMatmulReduceScatterTilingData *)tilingGM;
    m_ = tilingData->qbmmTilingdata.m;
    k_ = tilingData->qbmmTilingdata.k;
    n_ = tilingData->qbmmTilingdata.n;

    tpSizeM_ = m_ / tilingData->quantMatmulReduceScatterTilingInfo.rankSize;
    tpWorldSize_ = static_cast<int32_t>(tilingData->quantMatmulReduceScatterTilingInfo.rankSize);

    InitMatmulParams();

    if ASCEND_IS_AIV {
        TPipe pipe;
        tpipe_ = &pipe;
        if (tpipe_ != nullptr) {
            uint32_t paddedRowBytes =
                Blaze::Gemm::CeilDiv(static_cast<uint64_t>((tilingData->qbmmTilingdata.baseN) * sizeof(bfloat16_t)),
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

        yOutGM_.SetGlobalBuffer((__gm__ bfloat16_t *)yGM_);
    }
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
    KernelImpl kernel;
    kernel(params_);
}

template <typename X1Type, typename X2Type, typename ScaleType, bool isX2Nz>
__aicore__ inline void quantMatmulReduceScatterImpl<X1Type, X2Type, ScaleType, isX2Nz>::ReduceScatterProcess()
{
    BlockScheduler bs(params_.problemShape, params_.schParams);

    BlockCoord blockIdx;
    while (bs.GetTileIdx(blockIdx)) {
        int64_t mPos = 0;
        int64_t nPos = 0;
        bs.GetTileCoord(blockIdx, mPos, nPos);
        BlockShape singleShape =
            bs.template GetBlockShape<QuantMode::MX_PERGROUP_MODE, QuantMode::MX_PERGROUP_MODE, weightNz>(blockIdx);
        if (AscendC::Te::Get<IDX_M_TILEIDX>(singleShape) <= 0 || AscendC::Te::Get<IDX_N_TILEIDX>(singleShape) <= 0) {
            break;
        }
        CrossCoreWaitFlag(CROSS_CORE_FLAG_ID_NINE);
        // CommProcess(mPos, nPos, singleShape);
        CommProcessTensorMode(mPos, nPos, singleShape);
    }

    SyncAll<true>();
    WriteStatusToWin();
    ReadStatus();
    SyncAll<true>();

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
    // CommProcess 完成的是单个核的matmul计算结果搬入本卡ub 后，按rankid搬到对卡winGM的动作（localGM->UB + UB->RemoteGM）
    // 取到单核单次计算完成的matmul结果，basem x basen
    int64_t singleM = AscendC::Te::Get<IDX_M_TILEIDX>(singleShape);
    int64_t singleN = AscendC::Te::Get<IDX_N_TILEIDX>(singleShape);
    if (singleM <= 0 || singleN <= 0) {
        return;
    }

    int64_t tpSizeM = tpSizeM_; // scatter 后的每张卡m
    int64_t aivIdx = coreVid_ % 2;
    int64_t subTileCnt = Blaze::Gemm::CeilDiv(singleM, AIV_UB_TILE_M); // 每32行（m方向切）一个tile搬运

    __gm__ TypeC *workspaceAddr = reinterpret_cast<__gm__ TypeC *>(params_.mmadParams.cGmAddr);
    AscendC::GlobalTensor<bfloat16_t> workspaceGm;
    workspaceGm.SetGlobalBuffer(workspaceAddr);

    uint32_t blkLenBytes = static_cast<uint32_t>(singleN * sizeof(bfloat16_t)); // 单行（base n 的数据bytes）
    uint32_t rowPaddedBytes = Blaze::Gemm::CeilDiv(static_cast<uint64_t>(blkLenBytes), UB_ALIGN_BYTES) * UB_ALIGN_BYTES; // padding
    uint32_t rowInVECINElems = rowPaddedBytes / sizeof(bfloat16_t); // 单行搬运的元素个数

    for (int64_t st = aivIdx; st < subTileCnt; st += 2) {
        int64_t subTileM = Min(AIV_UB_TILE_M, singleM - st * AIV_UB_TILE_M);
        if (subTileM <= 0) {
            continue;
        }
        int64_t globalBaseM = mPos + st * AIV_UB_TILE_M; // for循环中对单核m tile的GM起始搬运行

        DataCopyPadExtParams<bfloat16_t> padParams{false, 0, 0, 0};
        AscendC::LocalTensor<bfloat16_t> batchLocal = inQueue_.AllocTensor<bfloat16_t>();

        DataCopyExtParams readParams{static_cast<uint16_t>(subTileM), blkLenBytes,
                                     static_cast<uint32_t>((n_ - singleN) * sizeof(bfloat16_t)), 0, 0};
        DataCopyPad(batchLocal, workspaceGm[globalBaseM * n_ + nPos], readParams, padParams); // 从GM搬运 subTileM * singleN 的数据块到ub上连续放置
        inQueue_.EnQue(batchLocal);
        batchLocal = inQueue_.DeQue<bfloat16_t>();

        // 本端UB--> 对端GM
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
__aicore__ inline void
quantMatmulReduceScatterImpl<X1Type, X2Type, ScaleType, isX2Nz>::CommProcessTensorMode(int64_t mPos, int64_t nPos,
                                                                             const BlockShape &singleShape)
{
    // CommProcess 完成的是单个核的matmul计算结果搬入本卡ub 后，按rankid搬到对卡winGM的动作（localGM->UB + UB->RemoteGM）
    // 取到单核单次计算完成的matmul结果，basem x basen
    int64_t singleM = AscendC::Te::Get<IDX_M_TILEIDX>(singleShape);
    int64_t singleN = AscendC::Te::Get<IDX_N_TILEIDX>(singleShape);
    if (singleM <= 0 || singleN <= 0) {
        return;
    }

    int64_t tpSizeM = tpSizeM_; // scatter 后的每张卡m
    int64_t aivIdx = coreVid_ % 2;
    int64_t subTileCnt = Blaze::Gemm::CeilDiv(singleM, AIV_UB_TILE_M); // 每32行（m方向切）一个tile搬运

    // __gm__ TypeC *workspaceAddr = reinterpret_cast<__gm__ TypeC *>(params_.mmadParams.cGmAddr);
    // AscendC::GlobalTensor<bfloat16_t> workspaceGm;
    // workspaceGm.SetGlobalBuffer(workspaceAddr);

    // uint32_t blkLenBytes = static_cast<uint32_t>(singleN * sizeof(bfloat16_t)); // 单行（base n 的数据bytes）
    // uint32_t rowPaddedBytes = Blaze::Gemm::CeilDiv(static_cast<uint64_t>(blkLenBytes), UB_ALIGN_BYTES) * UB_ALIGN_BYTES; // padding
    // uint32_t rowInVECINElems = rowPaddedBytes / sizeof(bfloat16_t); // 单行搬运的元素个数
    // 全局gm, 这个应该可以移到init只做一次
    auto layoutTensorC = AscendC::Te::FrameLayoutFormat<LayoutC, TypeC>{}(AscendC::Te::Get<MNK_M>(params_.problemShape), AscendC::Te::Get<MNK_N>(params_.problemShape));
    auto gmC = AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(cGmAddr_), layoutTensorC);
    auto copyGM2UB = AscendC::Te::MakeCopy(AscendC::Te::CopyGM2UB{}); // 构造拷贝数据通路类型
    auto copyUB2GM = AscendC::Te::MakeCopy(AscendC::Te::CopyUB2GM{}); // 构造拷贝数据通路类型

    // UB的padding
    uint64_t rowPaddedBytes = Blaze::Gemm::CeilDiv(static_cast<uint64_t>(singleN * sizeof(TypeC)), UB_ALIGN_BYTES) * UB_ALIGN_BYTES;
    uint64_t paddingN = rowPaddedBytes / sizeof(TypeC); // 单行搬运的元素个数

    for (int64_t st = aivIdx; st < subTileCnt; st += 2) {
        int64_t subTileM = Min(AIV_UB_TILE_M, singleM - st * AIV_UB_TILE_M);
        if (subTileM <= 0) {
            continue;
        }
        int64_t globalBaseM = mPos + st * AIV_UB_TILE_M; // for循环中对单核m tile的GM起始搬运行
        // 1. GM->UB
        auto layoutUB = MakeLayoutUB{}(subTileM, singleN); // 实际有效数据shape
        auto layoutPaddingUB = MakeLayoutUB{}(subTileM, paddingN); // padding 后的ub shape

        auto paddingUBTensor = AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::UB, TypeC>(0), layoutPaddingUB); // 构造padding UB tensor
        auto ubTensor = paddingUBTensor.Slice(AscendC::Te::MakeCoord(0, 0), AscendC::Te::MakeShape(subTileM, singleN)); // 切出有效数据 ub tensor
        // auto ubTensor = AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::UB, TypeC>(0), layoutUB); // 构造UB tensor
        auto gmTensor = gmC.Slice(AscendC::Te::MakeCoord(globalBaseM, nPos), AscendC::Te::MakeShape(subTileM, singleN)); // 切出当前tile 需要搬运的数据块

        AscendC::Te::Copy(copyGM2UB, ubTensor, gmTensor); // 后续的batchLocal 就变成ubTensor了

        SyncFunc<AscendC::HardEvent::MTE2_MTE3>(); // 确保UB上数据已经准备好了，再往外搬

        // DataCopyPadExtParams<bfloat16_t> padParams{false, 0, 0, 0};
        // AscendC::LocalTensor<bfloat16_t> batchLocal = inQueue_.AllocTensor<bfloat16_t>();

        // DataCopyExtParams readParams{static_cast<uint16_t>(subTileM), blkLenBytes,
        //                              static_cast<uint32_t>((n_ - singleN) * sizeof(bfloat16_t)), 0, 0};
        // DataCopyPad(batchLocal, workspaceGm[globalBaseM * n_ + nPos], readParams, padParams); // 从GM搬运 subTileM * singleN 的数据块到ub上连续放置
        // inQueue_.EnQue(batchLocal);
        // batchLocal = inQueue_.DeQue<bfloat16_t>();

        // 2. UB->GM
        // padding的问题后续讨论一下，先写一版不考虑32字节不对齐情况的
        // 本端UB--> 对端GM
        int64_t processed = 0;
        while (processed < subTileM) {
            int64_t globalRow = globalBaseM + processed;
            int32_t dstRankId = static_cast<int32_t>(globalRow / tpSizeM);
            if (dstRankId >= tpWorldSize_) {
                break;
            }
            int64_t tpBoundary = (dstRankId + 1) * tpSizeM;
            int64_t rowsInBatch = Min(subTileM - processed, tpBoundary - globalRow); // 当前while需要搬运的行数
            if (rowsInBatch <= 0) {
                break;
            }

            GM_ADDR remoteWinAddr = GetWindAddrByRankId(dstRankId);
            auto layoutTensorRemote = AscendC::Te::FrameLayoutFormat<LayoutC, TypeC>{}(
                tpSizeM, 
                AscendC::Te::Get<MNK_N>(params_.problemShape)); // scatter 到每张卡的m * 完整n
            __gm__ TypeC* remoteGmAddr_ = reinterpret_cast<__gm__ TypeC*>(remoteWinAddr);
            auto gmRemote = AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(remoteGmAddr_), layoutTensorRemote); // 对卡完整矩阵(tpSizeM * N)

            // 切出当前tile 需要搬运的数据块
            auto remoteGMTensor = gmRemote.Slice(
                AscendC::Te::MakeCoord(static_cast<uint64_t>(rankId_) * tpSizeM + globalRow % tpSizeM, nPos), 
                AscendC::Te::MakeShape(rowsInBatch, singleN)
            );
            auto ubProcessTensor = ubTensor.Slice(
                AscendC::Te::MakeCoord(processed, 0),
                AscendC::Te::MakeShape(rowsInBatch, singleN)
            );
            AscendC::Te::Copy(copyUB2GM, remoteGMTensor, ubProcessTensor);

            // AscendC::GlobalTensor<bfloat16_t> remoteGm;
            // remoteGm.SetGlobalBuffer((__gm__ bfloat16_t *)remoteWinAddr);
            // uint64_t dstBaseOffset = static_cast<uint64_t>(rankId_) * tpSizeM * n_ + (globalRow % tpSizeM) * n_ + nPos;

            // DataCopyExtParams writeParams{static_cast<uint16_t>(rowsInBatch), blkLenBytes, 0,
            //                               static_cast<uint32_t>((n_ - singleN) * sizeof(bfloat16_t)), 0};
            // DataCopyPad(remoteGm[dstBaseOffset], batchLocal[processed * rowInVECINElems], writeParams);

            processed += rowsInBatch;
        }
        SyncFunc<AscendC::HardEvent::MTE3_MTE2>(); // 确保当前UB->GM的动作完成
        //inQueue_.FreeTensor(batchLocal);
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