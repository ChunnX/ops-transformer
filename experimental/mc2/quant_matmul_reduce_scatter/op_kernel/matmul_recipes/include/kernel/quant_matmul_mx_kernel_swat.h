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
 * \file quant_matmul_mx_kernel_swat.h
 * \brief Kernel-side SWAT MX implementation for the non-full-load path.
 */

#pragma once

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#include "kernel_operator_intf.h"
#endif

#include "kernel_utils/common_utils.h"
#include "include/tensor_api/tensor.h"

#include "../block/quant_matmul_mx_block_mmad_swat.h"
#include "../block/quant_matmul_mx_block_scheduler_swat.h"
#include "../utils/constant.h"
#include "adv_api/hccl/hccl.h"
#include "../../common/op_kernel/moe_distribute_base.h"

namespace Kernel {
// Keep the class template parameter list in one place so the declaration and
// out-of-line member definitions stay perfectly aligned.
#define QBMM_MX_KERNEL_NO_FULL_LOAD_CLASS_TEM_PARAMS \
    template <class ProblemShape, class BlockMmad, class BlockScheduler>
#define QBMM_MX_KERNEL_NO_FULL_LOAD_FUN_TEM_PARAMS ProblemShape, BlockMmad, BlockScheduler

using namespace AscendC;

// HCCL window layout constants: data zone layout inside the 0/1 ping-pong
// windows. These match the values used by the host-side HCCL configuration.
constexpr static uint64_t WIN_DATA_INNER_OFFSET = 30UL * 1024UL * 1024UL;
constexpr static uint64_t SINGLE_ZERONE_DATA_ZONE_SIZE = 100UL * 1024UL * 1024UL;
constexpr static uint64_t UB_ALIGN_BYTES = 32UL;

// Sub-tile M dimension for AIV UB processing. Each scheduler tile is split
// into sub-tiles of this many rows; V0 and V1 alternate sub-tile indices.
constexpr static int64_t AIV_UB_TILE_M = 32;
constexpr uint32_t FLOAT_UB_ALIGN_NUM = 8U;
constexpr uint32_t STATE_ALIGN_BYTES = 32U;
constexpr uint32_t REDUCE_BUF_BYTES = 256U;
constexpr uint32_t BUFFER_NUM = 1U;

constexpr float EPSILON = 1e-6f;
constexpr float ONE = 1;

template <AscendC::HardEvent event>
__aicore__ inline void SyncFunc() {
    int32_t eventID = static_cast<int32_t>(GetTPipePtr()->FetchEventID(event));
    AscendC::SetFlag<event>(eventID);
    AscendC::WaitFlag<event>(eventID);
}

QBMM_MX_KERNEL_NO_FULL_LOAD_CLASS_TEM_PARAMS
class QuantMatmulMxKernelSwat {
public:
    __aicore__ inline QuantMatmulMxKernelSwat()
    {}
    __aicore__ inline ~QuantMatmulMxKernelSwat()
    {}

    static constexpr bool weightNz = BlockMmad::weightNz;
    static constexpr bool transA = BlockMmad::transA;
    static constexpr bool transB = BlockMmad::transB;
    using AType = typename BlockMmad::AType;
    using BType = typename BlockMmad::BType;
    using ScaleAType = typename BlockMmad::ScaleAType;
    using ScaleBType = typename BlockMmad::ScaleBType;
    using CType = typename BlockMmad::CType;

    // The kernel layer does not own a scheduling policy itself; it selects the
    // concrete scheduler from the block pipeline traits.
    using BlockSchedulerOp =
        typename Block::BlockSchedulerSelector<ProblemShape, BlockScheduler, transA, transB, AType>::SchedulerOp;

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

    struct QBMMTiling {
        // Base tile shape and L0C buffering mode selected by host tiling.
        uint32_t baseM;
        uint32_t baseN;
        uint32_t baseK;
        uint8_t dbL0C;
    };

    // Communication parameters for AIV alltoall path.
    struct CommParams {
        // Size of one TP domain = m / tpWorldSize.
        int64_t tpSizeM;
        // Number of TP ranks participating in the alltoall.
        int32_t tpWorldSize;
    };

    struct OutputParams {
        GM_ADDR y1Out = nullptr;
        GM_ADDR y2Out = nullptr;
        GM_ADDR xOut = nullptr;
        GM_ADDR y = nullptr;
        GM_ADDR gamma = nullptr;
    };

    struct Params {
        // Translate the serialized tiling payload once at the launcher
        // boundary, then keep the kernel, scheduler, and block MMAD layers on
        // their own typed parameter bundles.
        ProblemShape problemShape;
        BlockMmadParams mmadParams;
        L1Params l1Params;
        BlockSchedulerParams schParams;
        QBMMTiling qbmmParams;
        CommParams commParams;
        OutputParams outputParams;
    };

public:
    // Launch one SWAT kernel instance on the current AIC block.
    __aicore__ inline void operator()(const Params& params);

private:
    // Bind raw GM addresses to typed kernel pointers once per launch.
    __aicore__ inline void ResetGmAddr(const Params& params);

    // Wrap the full GM tensors and repeatedly slice them into per-block views
    // according to the scheduler output.
    __aicore__ inline void Process(const Params& params, BlockSchedulerOp& bs);

    template <typename TensorB, typename TensorScaleB>
    __aicore__ inline void SetL2Cache(
        const ProblemShape& problemShape, uint64_t curBaseM, uint64_t baseN, TensorB& gmB, TensorScaleB& gmScaleB);

    // `ProblemShape` stays generic at the template boundary, but the block
    // MMAD operator consumes a fixed 3D tuple internally.
    __aicore__ inline TupleShape ToShapeTuple(const ProblemShape& problemShape)
    {
        return {problemShape.m, problemShape.n, problemShape.k};
    }

    // AIV: read bf16 workspace and write to remote HCCL windows (alltoall).
    __aicore__ inline void CommProcess(
        int64_t mPos, int64_t nPos, const BlockShape& singleShape, const Params& params);

    __aicore__ inline void AddRmsNormAddCompute(uint32_t tokenIndex, uint32_t numCol,
                                                LocalTensor<float>& x1TmpFloatLocal,
                                                LocalTensor<float>& x2TmpFloatLocal,
                                                LocalTensor<float>& addOutTmpFloatLocal,
                                                const DataCopyExtParams& copyExtParams,
                                                const DataCopyPadExtParams<bfloat16_t>& copyPadExtParams);
    __aicore__ inline void AddRmsNormRmsNormCompute(uint32_t tokenIndex, uint32_t numCol,
                                                    LocalTensor<float>& xFp32, LocalTensor<float>& sqx,
                                                    LocalTensor<float>& gammaLocal,
                                                    const DataCopyExtParams& copyExtParams);

    __aicore__ inline void WriteStatusToWin();
    __aicore__ inline void ReadStatus();
    __aicore__ inline void ReadRemoteDataAdd(const Params& params);
    __aicore__ inline GM_ADDR GetWindStateAddrByRankId(const int32_t rankId);
    __aicore__ inline GM_ADDR GetWindAddrByRankId(const int32_t rankId);
    __aicore__ inline void SplitToCore(const uint32_t curSendCnt, const uint32_t curUseAivNum, const uint32_t coreId, uint32_t &startId, uint32_t &endId, uint32_t &sendNum);

private:
    BlockMmad mmadOp_;
    __gm__ AType* aGmAddr_;
    __gm__ BType* bGmAddr_;
    __gm__ CType* cGmAddr_;
    __gm__ fp8_e8m0_t* scaleAGmAddr_;
    __gm__ fp8_e8m0_t* scaleBGmAddr_;
    __gm__ float* y1GmAddr_;
    __gm__ bfloat16_t* y2GmAddr_;
    __gm__ bfloat16_t* xGmAddr_;
    __gm__ bfloat16_t* yGmAddr_;
    __gm__ float* gammaGmAddr_;
    __gm__ Mc2Kernel::HcclOpParam *winContext_{nullptr};
    TPipe *tpipe_{nullptr};
    uint32_t rankId_{0};
    uint32_t coreVid_{0};
    uint32_t tpWorldSize_{0};

    uint32_t m_;
    uint32_t n_;
    uint32_t k_;

    float armAvgFactor_;
    float epsilon_;

    TQueBind<TPosition::VECIN, TPosition::VECOUT,BUFFER_NUM> inQueue_;
    // TQueBind<TPosition::VECIN, TPosition::VECOUT, BUFFER_NUM> tokenQueue_;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> tokenQueue_;
    TBuf<> writeStateBuf_;
    TBuf<> readStateBuf_;
    TBuf<> resetStateBuf_;
    TBuf<> sumFp32Buf_;
    TBuf<> tokenFp32Buf_;

    // addrmsnormcast
    TBuf<> tokenBuf_;
    TBuf<> gammaBuf_;
    TBuf<> gammaFp32Buf_;
    TBuf<> rowTmpFloatBuf_;
    TBuf<> mulBuf_;
    TBuf<> resFp32Buf_;
    TBuf<TPosition::VECCALC> reduceFp32Buf_;

    GlobalTensor<bfloat16_t> yGM_;
    GlobalTensor<float> gammaGM_;
    GlobalTensor<float> y1OutGM_;
    GlobalTensor<bfloat16_t> y2OutGM_;
    GlobalTensor<bfloat16_t> xOutGM_;

    LocalTensor<bfloat16_t> tokenTensor_;
    LocalTensor<float> rowTmpFloatLocal_;
    LocalTensor<float> mulBufLocal_;
    
};

QBMM_MX_KERNEL_NO_FULL_LOAD_CLASS_TEM_PARAMS
__aicore__ inline void QuantMatmulMxKernelSwat<QBMM_MX_KERNEL_NO_FULL_LOAD_FUN_TEM_PARAMS>::operator()(
    const Params& params)
{
    ResetGmAddr(params);
    BlockSchedulerOp bs(params.problemShape, params.schParams);
    if ASCEND_IS_AIC {
        TupleShape problemShape_ = ToShapeTuple(params.problemShape);
        BlockShape l0TileShape{params.qbmmParams.baseM, params.qbmmParams.baseN, params.qbmmParams.baseK, 0};
        bool enableL0cPingPong = (params.qbmmParams.dbL0C > 1);
        mmadOp_.Init(problemShape_, l0TileShape, params.l1Params, enableL0cPingPong);
        Process(params, bs);
        return;
    }

    if ASCEND_IS_AIV {
        TPipe pipe;
        tpipe_ = &pipe;
        if (tpipe_ != nullptr) {
            uint32_t paddedRowBytes = ::CeilDiv(
                static_cast<uint64_t>(params.qbmmParams.baseN) * sizeof(bfloat16_t),
                UB_ALIGN_BYTES) * UB_ALIGN_BYTES;
            tpipe_->InitBuffer(inQueue_, BUFFER_NUM, AIV_UB_TILE_M * paddedRowBytes);
            tpipe_->InitBuffer(writeStateBuf_, STATE_ALIGN_BYTES);
            tpipe_->InitBuffer(readStateBuf_, STATE_ALIGN_BYTES);
            tpipe_->InitBuffer(resetStateBuf_, STATE_ALIGN_BYTES);
        }
        winContext_ = (__gm__ Mc2Kernel::HcclOpParam*)AscendC::GetHcclContext<HCCL_GROUP_ID_0>();
        rankId_ = Mc2Kernel::GetRankId(winContext_);
        coreVid_ = AscendC::GetBlockIdx();
        m_ = params.problemShape.m;
        n_ = params.problemShape.n;
        tpWorldSize_ = params.commParams.tpWorldSize;
        armAvgFactor_ = ONE / n_;
        epsilon_ = EPSILON;
        // PRINTF("[DBG] AIV rank=%u core=%u m=%u n=%u baseN=%u\n", rankId_, coreVid_, m_, n_, params.qbmmParams.baseN);
        // if (coreVid_ == 0) { PRINTF("[DBG_ND] rank%u core%u n_=%u m_=%u tpWorldSize=%u\n", rankId_, coreVid_, n_, m_, tpWorldSize_); }

        yGM_.SetGlobalBuffer(yGmAddr_);
        gammaGM_.SetGlobalBuffer(gammaGmAddr_);
        y1OutGM_.SetGlobalBuffer(y1GmAddr_);
        y2OutGM_.SetGlobalBuffer(y2GmAddr_);
        xOutGM_.SetGlobalBuffer(xGmAddr_);

        // uint32_t tileCount = 0;
        BlockCoord blockIdx;
        while (bs.template GetTileIdx<weightNz>(blockIdx)) {
            int64_t mPos = AscendC::Te::Get<MNK_M>(blockIdx);
            int64_t nPos = AscendC::Te::Get<MNK_N>(blockIdx);
            BlockShape singleShape = bs.template GetBlockShape<weightNz>(blockIdx);
            if (AscendC::Te::Get<MNK_M>(singleShape) <= 0 || AscendC::Te::Get<MNK_N>(singleShape) <= 0) {
                // Tail splitting can create empty logical slices; ignore them and
                // stop the current block once no useful work remains.
                break;
            }
            CrossCoreWaitFlag(9);
            // tileCount++;
            CommProcess(mPos, nPos, singleShape, params);
        }
        // PRINTF("[DBG] AIV rank=%u core=%u CommProcess done tiles=%u\n", rankId_, coreVid_, tileCount);

        // 状态去做全核同步
        SyncAll<true>();
        // PRINTF("[DBG] AIV rank=%u core=%u SyncAll1 done\n", rankId_, coreVid_);
        // 当前die的数据已经发完
        WriteStatusToWin();
        ReadStatus();
        // PRINTF("[DBG] AIV rank=%u core=%u WriteStatus+ReadStatus done\n", rankId_, coreVid_);
        SyncAll<true>();    //确保前4个核都等到了状态，即数据区完全ready
        // PRINTF("[DBG] AIV rank=%u core=%u SyncAll2 done\n", rankId_, coreVid_);
        
        tpipe_->Reset();
        tpipe_->InitBuffer(tokenQueue_, BUFFER_NUM, params.problemShape.n * sizeof(float));   //涉及到Cast的src和dst记得，小cast大要分配大的空间
        tpipe_->InitBuffer(sumFp32Buf_, params.problemShape.n * sizeof(float));
        tpipe_->InitBuffer(tokenFp32Buf_, params.problemShape.n * sizeof(float));

        tpipe_->InitBuffer(tokenBuf_, params.problemShape.n * sizeof(bfloat16_t));
        tpipe_->InitBuffer(rowTmpFloatBuf_, params.problemShape.n * sizeof(float));
        tpipe_->InitBuffer(mulBuf_, params.problemShape.n * sizeof(float));
        tpipe_->InitBuffer(reduceFp32Buf_, REDUCE_BUF_BYTES);
        tpipe_->InitBuffer(resFp32Buf_, params.problemShape.n * sizeof(float));
        tpipe_->InitBuffer(gammaBuf_, params.problemShape.n * sizeof(float));
        tpipe_->InitBuffer(gammaFp32Buf_, params.problemShape.n * sizeof(float));

        ReadRemoteDataAdd(params);
        // PRINTF("[DBG] AIV rank=%u core=%u ReadRemoteDataAdd done\n", rankId_, coreVid_);
        return;
    }
}

QBMM_MX_KERNEL_NO_FULL_LOAD_CLASS_TEM_PARAMS
__aicore__ inline void QuantMatmulMxKernelSwat<QBMM_MX_KERNEL_NO_FULL_LOAD_FUN_TEM_PARAMS>::ResetGmAddr(
    const Params& params)
{
    // The sample launcher passes raw GM addresses through `Params`; convert
    // them here once so the hot loop works with typed pointers only.
    aGmAddr_ = reinterpret_cast<__gm__ AType*>(params.mmadParams.aGmAddr);
    bGmAddr_ = reinterpret_cast<__gm__ BType*>(params.mmadParams.bGmAddr);
    cGmAddr_ = reinterpret_cast<__gm__ CType*>(params.mmadParams.cGmAddr);
    scaleAGmAddr_ = reinterpret_cast<__gm__ fp8_e8m0_t*>(params.mmadParams.scaleAGmAddr);
    scaleBGmAddr_ = reinterpret_cast<__gm__ fp8_e8m0_t*>(params.mmadParams.scaleBGmAddr);
    y1GmAddr_ = reinterpret_cast<__gm__ float*>(params.outputParams.y1Out);
    y2GmAddr_ = reinterpret_cast<__gm__ bfloat16_t*>(params.outputParams.y2Out);
    xGmAddr_ = reinterpret_cast<__gm__ bfloat16_t*>(params.outputParams.xOut);
    yGmAddr_ = reinterpret_cast<__gm__ bfloat16_t*>(params.outputParams.y);
    gammaGmAddr_ = reinterpret_cast<__gm__ float*>(params.outputParams.gamma);
}

QBMM_MX_KERNEL_NO_FULL_LOAD_CLASS_TEM_PARAMS
template <typename TensorB, typename TensorScaleB>
__aicore__ inline void QuantMatmulMxKernelSwat<QBMM_MX_KERNEL_NO_FULL_LOAD_FUN_TEM_PARAMS>::SetL2Cache(
    const ProblemShape& problemShape, uint64_t curBaseM, uint64_t baseN, TensorB& gmB, TensorScaleB& gmScaleB)
{
    if constexpr (weightNz) {
        if (curBaseM >= problemShape.m) {
            gmB.SetL2CacheHint(AscendC::Te::CacheMode::CACHE_MODE_DISABLE);
            gmScaleB.SetL2CacheHint(AscendC::Te::CacheMode::CACHE_MODE_DISABLE);
        } else {
            gmB.SetL2CacheHint(AscendC::Te::CacheMode::CACHE_MODE_NORMAL);
            gmScaleB.SetL2CacheHint(AscendC::Te::CacheMode::CACHE_MODE_NORMAL);
        }
    } else {
        if constexpr (transB) {
            if (curBaseM >= problemShape.m && (problemShape.k & 0xff) == 0) {
                gmB.SetL2CacheHint(AscendC::Te::CacheMode::CACHE_MODE_DISABLE);
                gmScaleB.SetL2CacheHint(AscendC::Te::CacheMode::CACHE_MODE_DISABLE);
            } else {
                gmB.SetL2CacheHint(AscendC::Te::CacheMode::CACHE_MODE_NORMAL);
                gmScaleB.SetL2CacheHint(AscendC::Te::CacheMode::CACHE_MODE_NORMAL);
            }
        } else {
            if (curBaseM >= problemShape.m && (problemShape.n & 0xff) == 0 && (baseN & 0xff) == 0) {
                gmB.SetL2CacheHint(AscendC::Te::CacheMode::CACHE_MODE_DISABLE);
                gmScaleB.SetL2CacheHint(AscendC::Te::CacheMode::CACHE_MODE_DISABLE);
            } else {
                gmB.SetL2CacheHint(AscendC::Te::CacheMode::CACHE_MODE_NORMAL);
                gmScaleB.SetL2CacheHint(AscendC::Te::CacheMode::CACHE_MODE_NORMAL);
            }
        }
    }
}

QBMM_MX_KERNEL_NO_FULL_LOAD_CLASS_TEM_PARAMS
__aicore__ inline void QuantMatmulMxKernelSwat<QBMM_MX_KERNEL_NO_FULL_LOAD_FUN_TEM_PARAMS>::Process(
    const Params& params, BlockSchedulerOp& bs)
{
     // Build full-GM tensor views once, then slice them per scheduled block.
    int64_t kScaleSize =
        ::CeilDiv(static_cast<int64_t>(params.problemShape.k), static_cast<int64_t>(MXFP_DIVISOR_SIZE)) *
        MXFP_MULTI_BASE_SIZE;
    auto layoutA = MakeLayoutA{}(params.problemShape.m, params.problemShape.k);
    auto layoutScaleA = MakeLayoutScaleA{}(params.problemShape.m, kScaleSize);
    auto layoutB = MakeLayoutB{}(params.problemShape.k, params.problemShape.n);
    auto layoutScaleB = MakeLayoutScaleB{}(kScaleSize, params.problemShape.n);
    auto layoutC =
        AscendC::Te::MakeFrameLayout<AscendC::Te::NDExtLayoutPtn>(params.problemShape.m, params.problemShape.n);

    auto gmA = AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(aGmAddr_), layoutA);
    auto gmScaleA =
        AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(scaleAGmAddr_), layoutScaleA);
    auto gmB = AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(bGmAddr_), layoutB);
    auto gmScaleB =
        AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(scaleBGmAddr_), layoutScaleB);
    auto gmC = AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(cGmAddr_), layoutC);

    BlockCoord blockIdx;
    constexpr int64_t kPos = 0L;
    SetL2Cache(params.problemShape, params.qbmmParams.baseM, params.qbmmParams.baseN, gmB, gmScaleB);
    while (bs.template GetTileIdx<weightNz>(blockIdx)) {
        // The scheduler packs GM origin into M/N and retains logical tile
        // indices in K/B so shape reconstruction still works.
        int64_t mPos = AscendC::Te::Get<MNK_M>(blockIdx);
        int64_t nPos = AscendC::Te::Get<MNK_N>(blockIdx);
        BlockShape singleShape = bs.template GetBlockShape<weightNz>(blockIdx);
        if (AscendC::Te::Get<MNK_M>(singleShape) <= 0 || AscendC::Te::Get<MNK_N>(singleShape) <= 0) {
            // Tail splitting can create empty logical slices; ignore them and
            // stop the current block once no useful work remains.
            return;
        }

        // `blockIdx` now carries both GM origin and logical tile metadata.
        auto gmBlockA =
            gmA.Slice(AscendC::Te::MakeCoord(mPos, kPos), AscendC::Te::MakeShape(AscendC::Te::Get<MNK_M>(singleShape), params.problemShape.k));
        auto gmBlockScaleA =
            gmScaleA.Slice(AscendC::Te::MakeCoord(mPos, kPos), AscendC::Te::MakeShape(AscendC::Te::Get<MNK_M>(singleShape), kScaleSize));
        auto gmBlockB =
            gmB.Slice(AscendC::Te::MakeCoord(kPos, nPos), AscendC::Te::MakeShape(params.problemShape.k, AscendC::Te::Get<MNK_N>(singleShape)));
        auto gmBlockScaleB =
            gmScaleB.Slice(AscendC::Te::MakeCoord(kPos, nPos), AscendC::Te::MakeShape(kScaleSize, AscendC::Te::Get<MNK_N>(singleShape)));
        auto gmBlockC =
            gmC.Slice(AscendC::Te::MakeCoord(mPos, nPos),
                AscendC::Te::MakeShape(AscendC::Te::Get<MNK_M>(singleShape), AscendC::Te::Get<MNK_N>(singleShape)));

        // The block MMAD layer owns all data movement below GM granularity and
        // performs the actual accumulation for this scheduled tile.
        mmadOp_(gmBlockA, gmBlockB, gmBlockScaleA, gmBlockScaleB, gmBlockC, singleShape);
        CrossCoreSetFlag<0x2, PIPE_FIX>(9);
    }
}

QBMM_MX_KERNEL_NO_FULL_LOAD_CLASS_TEM_PARAMS
__aicore__ inline GM_ADDR QuantMatmulMxKernelSwat<QBMM_MX_KERNEL_NO_FULL_LOAD_FUN_TEM_PARAMS>::GetWindAddrByRankId(const int32_t rankId)
{
    return Mc2Kernel::GetBaseWindAddrByRankId(winContext_, rankId, rankId_);
}


QBMM_MX_KERNEL_NO_FULL_LOAD_CLASS_TEM_PARAMS
__aicore__ inline GM_ADDR QuantMatmulMxKernelSwat<QBMM_MX_KERNEL_NO_FULL_LOAD_FUN_TEM_PARAMS>::GetWindStateAddrByRankId(const int32_t rankId)
{
    return Mc2Kernel::GetBaseWindStateAddrByRankId(winContext_, rankId, rankId_);
}

QBMM_MX_KERNEL_NO_FULL_LOAD_CLASS_TEM_PARAMS
__aicore__ inline void QuantMatmulMxKernelSwat<QBMM_MX_KERNEL_NO_FULL_LOAD_FUN_TEM_PARAMS>::SplitToCore(const uint32_t curSendCnt, const uint32_t curUseCoreNum, const uint32_t coreId, uint32_t &startId, uint32_t &endId, uint32_t &sendNum)
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

QBMM_MX_KERNEL_NO_FULL_LOAD_CLASS_TEM_PARAMS
__aicore__ inline void
QuantMatmulMxKernelSwat<QBMM_MX_KERNEL_NO_FULL_LOAD_FUN_TEM_PARAMS>::CommProcess(
    int64_t mPos, int64_t nPos, const BlockShape& singleShape, const Params& params)
{
    int64_t singleM = AscendC::Te::Get<MNK_M>(singleShape);
    int64_t singleN = AscendC::Te::Get<MNK_N>(singleShape);
    if (singleM <= 0 || singleN <= 0) {
        return;
    }

    // m_ = params.problemShape.m;
    // n_ = params.problemShape.n;
    int64_t tpSizeM = params.commParams.tpSizeM;
    // tpWorldSize_ = params.commParams.tpWorldSize;

    int64_t aivIdx = coreVid_ % 2;
    int64_t subTileCnt = ::CeilDiv(singleM, AIV_UB_TILE_M);

    __gm__ CType* workspaceAddr =
        reinterpret_cast<__gm__ CType*>(params.mmadParams.cGmAddr);
    GlobalTensor<bfloat16_t> workspaceGm;
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

        // 1) 整块读子 tile 到 VECIN（一次性 GM→VECIN）
        DataCopyPadExtParams<bfloat16_t> padParams{false, 0, 0, 0};
        LocalTensor<bfloat16_t> batchLocal = inQueue_.AllocTensor<bfloat16_t>();

        DataCopyExtParams readParams{
            static_cast<uint16_t>(subTileM),
            blkLenBytes,
            static_cast<uint32_t>((n_ - singleN) * sizeof(bfloat16_t)),
            0,
            0
        };
        DataCopyPad(batchLocal, workspaceGm[globalBaseM * n_ + nPos], readParams, padParams);
        inQueue_.EnQue(batchLocal);
        batchLocal = inQueue_.DeQue<bfloat16_t>();

        // 2) 按 TP domain 分批次写远端窗口（每批一次 VECIN→GM）
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

            GM_ADDR curWinAddr = GetWindAddrByRankId(rankId_);
            // printf("currank is %d, curwinaddr is %p. dstrankid is %d, dstwinaddr is %p\n", rankId_, curWinAddr, dstRankId, remoteWinAddr);
            // printf("currank is %d, dstrankid is %d\n", rankId_, dstRankId);

            GlobalTensor<bfloat16_t> remoteGm;
            remoteGm.SetGlobalBuffer((__gm__ bfloat16_t*)remoteWinAddr);
            uint64_t dstBaseOffset = static_cast<uint64_t>(rankId_) * tpSizeM * n_ +
                                     (globalRow % tpSizeM) * n_ + nPos;

            DataCopyExtParams writeParams{
                static_cast<uint16_t>(rowsInBatch),
                blkLenBytes,
                0,
                static_cast<uint32_t>((n_ - singleN) * sizeof(bfloat16_t)),
                0
            };
            DataCopyPad(remoteGm[dstBaseOffset], batchLocal[processed * rowInVECINElems], writeParams);

            processed += rowsInBatch;
        }

        inQueue_.FreeTensor(batchLocal);
    }
}

QBMM_MX_KERNEL_NO_FULL_LOAD_CLASS_TEM_PARAMS
__aicore__ inline void QuantMatmulMxKernelSwat<QBMM_MX_KERNEL_NO_FULL_LOAD_FUN_TEM_PARAMS>::AddRmsNormAddCompute(uint32_t tokenIndex, uint32_t numCol,
                                                                                LocalTensor<float>& x1TmpFloatLocal,
                                                                                LocalTensor<float>& x2TmpFloatLocal,
                                                                                LocalTensor<float>& addOutTmpFloatLocal,
                                                                                const DataCopyExtParams& copyExtParams,
                                                                                const DataCopyPadExtParams<bfloat16_t>& copyPadExtParams)
{
    // 计算x + residual_x
    LocalTensor<bfloat16_t> x2 = tokenBuf_.Get<bfloat16_t>();
    DataCopyPad(x2, yGM_[tokenIndex * n_], copyExtParams, copyPadExtParams);
    SyncFunc<AscendC::HardEvent::MTE2_V>();
    Cast(x2TmpFloatLocal, x2, AscendC::RoundMode::CAST_NONE, numCol);
    PipeBarrier<PIPE_V>();
    AscendC::Add(addOutTmpFloatLocal, x1TmpFloatLocal, x2TmpFloatLocal, numCol);
}

QBMM_MX_KERNEL_NO_FULL_LOAD_CLASS_TEM_PARAMS
__aicore__ inline void QuantMatmulMxKernelSwat<QBMM_MX_KERNEL_NO_FULL_LOAD_FUN_TEM_PARAMS>::AddRmsNormRmsNormCompute(uint32_t tokenIndex, uint32_t numCol,
                                                    LocalTensor<float>& xFp32, LocalTensor<float>& sqx,
                                                    LocalTensor<float>& gammaLocal,
                                                    const DataCopyExtParams& copyExtParams)
{
    // 计算rstd
    LocalTensor<float> reduceBufLocal = reduceFp32Buf_.Get<float>();
    LocalTensor<float> resFp32Tensor = resFp32Buf_.Get<float>();
    Mul(sqx, xFp32, xFp32, numCol);
    PipeBarrier<PIPE_V>();
    Muls(sqx, sqx, armAvgFactor_, numCol);
    PipeBarrier<PIPE_V>();
    ReduceSum(sqx, sqx, reduceBufLocal, numCol);
    PipeBarrier<PIPE_V>();
    Adds(sqx, sqx, epsilon_, 1);
    PipeBarrier<PIPE_V>();
    Sqrt(sqx, sqx, 1);
    Duplicate(reduceBufLocal, ONE, 1);
    PipeBarrier<PIPE_V>();
    Div(reduceBufLocal, reduceBufLocal, sqx, 1);
    // 计算y
    SyncFunc<AscendC::HardEvent::V_S>();
    float rstdValue = reduceBufLocal.GetValue(0);
    SyncFunc<AscendC::HardEvent::S_V>();
    Muls(xFp32, xFp32, rstdValue, numCol);
    PipeBarrier<PIPE_V>();
    LocalTensor<bfloat16_t> yLocal = rowTmpFloatBuf_.Get<bfloat16_t>();
    PipeBarrier<PIPE_V>();
    Mul(xFp32, xFp32, gammaLocal, numCol);
    PipeBarrier<PIPE_V>();
    Cast(yLocal, xFp32, RoundMode::CAST_RINT, numCol);
    // y结果搬出
    SyncFunc<AscendC::HardEvent::V_MTE3>();
    DataCopy(y2OutGM_[tokenIndex * n_], yLocal, n_);
    PipeBarrier<PIPE_V>();
    Cast(resFp32Tensor, yLocal, RoundMode::CAST_NONE, numCol);
    SyncFunc<AscendC::HardEvent::V_MTE3>();
    DataCopy(y1OutGM_[tokenIndex * n_], resFp32Tensor, n_);
}


QBMM_MX_KERNEL_NO_FULL_LOAD_CLASS_TEM_PARAMS
__aicore__ inline void QuantMatmulMxKernelSwat<QBMM_MX_KERNEL_NO_FULL_LOAD_FUN_TEM_PARAMS>::ReadRemoteDataAdd(const Params& params)
{
    uint32_t startRowId = 0;
    uint32_t endRowId = 0;
    uint32_t rowNum = 0;
    uint32_t tokenIndex = 0;
    SplitToCore(m_ / tpWorldSize_, GetBlockNum() * 2, coreVid_, startRowId, endRowId, rowNum);

    // PRINTF("[DBG] AIV rank=%u core=%u ReadRemote start=%u end=%u rows=%u\n", rankId_, coreVid_, startRowId, endRowId, rowNum);

    // PRINTF("startRowId is %d, endRowId is %d, rowNum is %d, tokenIndex is %d, vid is %d, m_ / tpWorldSize_ is %d\n", startRowId, endRowId, rowNum, tokenIndex, coreVid_, m_ / tpWorldSize_);

    LocalTensor<float> sumFp32Tensor = sumFp32Buf_.Get<float>();
    LocalTensor<float> tokenFp32Tensor = tokenFp32Buf_.Get<float>();
    mulBufLocal_ = mulBuf_.Get<float>();
    uint32_t curVidOffset = startRowId * n_;
    GlobalTensor<bfloat16_t> localWinTensor;
    localWinTensor.SetGlobalBuffer((__gm__ bfloat16_t*)GetWindAddrByRankId(rankId_));

    // PRINTF("aiv num is %d!\n", GetBlockNum() * 2);
    // DumpTensor(localWinTensor, 777, 3);
    // if (coreVid_ == 0) {
    //     DumpTensor(localWinTensor[80 * 512], 666, 10 * 512);
    // }

    DataCopyExtParams expandXCopyParams{1U, static_cast<uint32_t>(n_ * sizeof(bfloat16_t)), 0U, 0U, 0U};
    DataCopyExtParams gammaCopyParams{1U, static_cast<uint32_t>(n_ * sizeof(float)), 0U, 0U, 0U};
    const DataCopyPadExtParams<bfloat16_t> copyPadXTypeParams{false, 0U, 0U, 0U};
    const DataCopyPadExtParams<float> copyPadFloatParams{false, 0U, 0U, 0U};

    // GlobalTensor<bfloat16_t> xOutTensor;
    // // xGmAddr_
    // // xOutTensor.SetGlobalBuffer((__gm__ bfloat16_t*)params.outputParams.xOut);
    // xOutTensor.SetGlobalBuffer(xGmAddr_);

    for(uint32_t tileIdx = 0; tileIdx < rowNum; ++tileIdx) {
        Duplicate<float>(sumFp32Tensor, (float)0.0, n_);
        tokenIndex = startRowId + tileIdx;
        uint32_t curVidTileOffset = tokenIndex * n_;
        for(int tpIndex = 0; tpIndex < tpWorldSize_; ++tpIndex) {
            uint32_t curOffset = curVidTileOffset + tpIndex * (m_ / tpWorldSize_ * n_);
            tokenTensor_ = tokenQueue_.AllocTensor<bfloat16_t>();
            DataCopy(tokenTensor_, localWinTensor[curOffset], n_);
            tokenQueue_.EnQue(tokenTensor_);
            tokenTensor_ = tokenQueue_.DeQue<bfloat16_t>();
            SyncFunc<AscendC::HardEvent::MTE2_V>();
            Cast(tokenFp32Tensor, tokenTensor_, RoundMode::CAST_NONE, n_);
            PipeBarrier<PIPE_V>();
            Add(sumFp32Tensor, sumFp32Tensor, tokenFp32Tensor, n_);
            // DataCopy(xOutTensor[curOffset], tokenTensor_, n_);
            // DumpTensor(xOutTensor[curOffset], 888, n_);
            tokenQueue_.FreeTensor<bfloat16_t>(tokenTensor_);
            // PipeBarrier<PIPE_V>();
        }
        
        // ReduceScatter only — 不做 Add(residual y) / RmsNorm / Cast 后处理。
        // 把跨 rank Reduce 累加后的 fp32 直接 Cast 回 bf16 写到 xOutGM_。
        // y1 / y2 / y(residual) / gamma 这一路不参与计算，仅作为 OpDef 中
        // 占位输入输出存在；上层调用方对应位置可以传 nullptr / 0 字节缓冲。
        LocalTensor<bfloat16_t> sumBufLocal = tokenBuf_.Get<bfloat16_t>();
        PipeBarrier<PIPE_V>();
        Cast(sumBufLocal, sumFp32Tensor, AscendC::RoundMode::CAST_RINT, n_);
        SyncFunc<AscendC::HardEvent::V_MTE3>();
        DataCopyPad(xOutGM_[tokenIndex * n_], sumBufLocal, expandXCopyParams);
    }
}



QBMM_MX_KERNEL_NO_FULL_LOAD_CLASS_TEM_PARAMS
__aicore__ inline void
QuantMatmulMxKernelSwat<QBMM_MX_KERNEL_NO_FULL_LOAD_FUN_TEM_PARAMS>::WriteStatusToWin()
{
    if (coreVid_ >= tpWorldSize_) {
        return;
    }
    uint32_t curOffset = rankId_ * FLOAT_UB_ALIGN_NUM;
    // 写入状态到对端，每个核写一个状态到一个rank
    LocalTensor<float> statusTensor = writeStateBuf_.Get<float>();
    statusTensor(0) = (float)1;
    uint32_t curDstId = coreVid_;
    GM_ADDR remoteWinStateGM = GetWindStateAddrByRankId(curDstId);
    GlobalTensor<float> stateGMTensor;
    stateGMTensor.SetGlobalBuffer((__gm__ float*)remoteWinStateGM);
    SyncFunc<AscendC::HardEvent::S_MTE3>();
    DataCopy(stateGMTensor[curOffset], statusTensor, FLOAT_UB_ALIGN_NUM);
}

QBMM_MX_KERNEL_NO_FULL_LOAD_CLASS_TEM_PARAMS
__aicore__ inline void QuantMatmulMxKernelSwat<QBMM_MX_KERNEL_NO_FULL_LOAD_FUN_TEM_PARAMS>::ReadStatus()
{
    // 粗粒度状态区逻辑，只有tpwordsize个状态，所需的v核只需要4个
    if (coreVid_ >= tpWorldSize_) {
        return;
    }
    GM_ADDR stateGM = GetWindStateAddrByRankId(rankId_);
    GlobalTensor<float> selfStateWinTensor;
    // 获取当前核所需读取状态的头地址
    selfStateWinTensor.SetGlobalBuffer((__gm__ float*)(stateGM));
    uint32_t offset = coreVid_ * FLOAT_UB_ALIGN_NUM;
    LocalTensor<float> statusTensor = readStateBuf_.Get<float>();
    LocalTensor<float> resetStatusTensor = resetStateBuf_.Get<float>();
    Duplicate<float>(resetStatusTensor, 0.0, FLOAT_UB_ALIGN_NUM);
    float flag = -1;
    uint32_t statusCnt = FLOAT_UB_ALIGN_NUM;
    float minTarget = (float)0.5;
    float maxTarget = (float)1.5;
    // 读取statusCnt个数据求和
    while ((flag < minTarget) || (flag > maxTarget)) {
        SyncFunc<AscendC::HardEvent::S_MTE2>();
        DataCopy(statusTensor, selfStateWinTensor[offset], statusCnt);
        SyncFunc<AscendC::HardEvent::MTE2_S>();
        flag = statusTensor(0);
    }   
    // reset state
    SyncFunc<AscendC::HardEvent::V_MTE3>();
    SyncFunc<AscendC::HardEvent::MTE2_MTE3>();
    DataCopy(selfStateWinTensor[offset], resetStatusTensor, statusCnt);
}


} // namespace Kernel

#undef QBMM_MX_KERNEL_NO_FULL_LOAD_CLASS_TEM_PARAMS
#undef QBMM_MX_KERNEL_NO_FULL_LOAD_FUN_TEM_PARAMS

