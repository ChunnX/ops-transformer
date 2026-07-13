/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MEGA_MOE_SINGLE_CORE_H
#define MEGA_MOE_SINGLE_CORE_H

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "mega_moe_base.h"
#include "mega_moe_impl.h"
#if __has_include("../../moe_distribute_dispatch_v2/quantize_functions.h")
#include "../../moe_distribute_dispatch_v2/quantize_functions.h"
#else
#include "../../../moe_distribute_dispatch_v2/op_kernel/quantize_functions.h"
#endif

namespace MegaMoeImpl {

// AIV logical thread information supplied by the caller. threadIdx must be a
// zero-based, contiguous index in [0, threadNum), independent of physical AIV IDs.
struct AivThreadContext {
    uint32_t threadIdx;
    uint32_t threadNum;
};

struct MoeTopologyContext {
    uint32_t rankId;
    uint32_t worldSize;
    uint32_t expertPerRank;
};

struct TokenShapeContext {
    int32_t tokenNum;
    uint32_t topK;
    uint32_t tokenHiddenDim;
};

// Stable information shared by the dispatch preparation stage.
struct DispatchPrepareContext {
    AivThreadContext thread;
    MoeTopologyContext topology;
    TokenShapeContext tokenShape;
};

__aicore__ inline void TilingByThreadContext(int32_t totalLen, int32_t &threadLen, int32_t &threadOffset,
    uint32_t threadIdx, uint32_t threadNum, int32_t align = ALIGN_32)
{
    if (threadNum == 0U || threadIdx >= threadNum) {
        threadLen = 0;
        threadOffset = 0;
        return;
    }
    int32_t lenPerThread = Ops::Base::CeilDiv(static_cast<uint32_t>(totalLen), threadNum);
    int32_t lenPerThreadAlign =
        Ops::Base::CeilAlign(static_cast<uint32_t>(lenPerThread), static_cast<uint32_t>(align));
    threadLen = lenPerThreadAlign;
    threadOffset = static_cast<int32_t>(threadIdx) * lenPerThreadAlign;
    if (threadOffset + threadLen >= totalLen) {
        threadLen = totalLen - threadOffset;
    }
    if (threadOffset >= totalLen) {
        threadLen = 0;
    }
}

struct ResetWorkspaceArgs {
    GM_ADDR flagSwiGluToGmm2Ptr;
    GM_ADDR gmm2CombineSyncCounterPtr;
    int32_t flagNum;
    int32_t gmm2CombineSyncCounterNum;
    bool resetCombineCounters;
};

__aicore__ inline void ResetGmm2CombineSyncCountersOneCore(const AivThreadContext &thread,
    const ResetWorkspaceArgs &args, LocalTensor<int32_t> &resetTensor)
{
    if constexpr (g_coreType == AIV) {
        if (thread.threadNum == 0U || thread.threadIdx >= thread.threadNum) {
            return;
        }
        int32_t threadLen, threadOffset;
        TilingByThreadContext(args.gmm2CombineSyncCounterNum, threadLen, threadOffset,
            thread.threadIdx, thread.threadNum);
        GlobalTensor<int32_t> gmm2CombineSyncCounterGm;
        gmm2CombineSyncCounterGm.SetGlobalBuffer((__gm__ int32_t*)args.gmm2CombineSyncCounterPtr);
        if (threadLen > 0) {
            Duplicate(resetTensor, 0, threadLen);
            SyncFuncStatic<AscendC::HardEvent::V_MTE3, SYNC_EVENT_ID2>();
            DataCopy(gmm2CombineSyncCounterGm[threadOffset], resetTensor, threadLen);
        }
    }
}

__aicore__ inline void ResetFlagListOneCore(const DispatchPrepareContext &context,
    const ResetWorkspaceArgs &args, LocalTensor<int32_t> &resetTensor)
{
    if constexpr (g_coreType == AIC) {
        return;
    }
    const AivThreadContext &thread = context.thread;
    if (thread.threadNum == 0U || thread.threadIdx >= thread.threadNum) {
        return;
    }
    GlobalTensor<int32_t> swigluToGmm2FlagGm;
    swigluToGmm2FlagGm.SetGlobalBuffer((__gm__ int32_t*)args.flagSwiGluToGmm2Ptr);
    int32_t threadLen, threadOffset;
    TilingByThreadContext(args.flagNum, threadLen, threadOffset, thread.threadIdx, thread.threadNum, 1);
    DataCopyExtParams rankSyncCopyParams{1U, static_cast<uint32_t>(threadLen * sizeof(int32_t)), 0U, 0U, 0U};
    SyncFuncStatic<AscendC::HardEvent::V_MTE3, SYNC_EVENT_ID2>();
    if (threadLen != 0) {
        DataCopyPad(swigluToGmm2FlagGm[threadOffset], resetTensor, rankSyncCopyParams);
    }
    if (args.resetCombineCounters) {
        ResetGmm2CombineSyncCountersOneCore(thread, args, resetTensor);
    }
}

struct SendMaskArgs {
    GM_ADDR expertIdxGmAddr;
    GM_ADDR *winRankAddr;
    int32_t compareCount;
    uint32_t maskAlignSize;
    uint32_t maskSlotSize;
    uint64_t maskWinOffset;
};

struct SendMaskScratch {
    LocalTensor<int32_t> topkIdsTensor;
    LocalTensor<uint8_t> sendMaskTensor[DOUBLE_BUFFER];
    LocalTensor<int32_t> sendGatherOutTensor;
};

__aicore__ inline LocalTensor<uint8_t> GetSendMaskTensor(SendMaskScratch &scratch, int32_t iter)
{
    return scratch.sendMaskTensor[iter % static_cast<int32_t>(DOUBLE_BUFFER)];
}

__aicore__ inline void SendMaskCalOneExpert(const DispatchPrepareContext &context,
    const SendMaskArgs &args, SendMaskScratch &scratch, uint64_t sendTotalNum,
    int32_t curExpertId, int32_t iter)
{
    if constexpr (g_coreType == AIC) {
        return;
    }

    constexpr TEventID kBufEvents[DOUBLE_BUFFER] = {EVENT_ID0, EVENT_ID1};
    uint32_t countWordIdx = static_cast<uint32_t>(args.maskAlignSize) / sizeof(int32_t);
    DataCopyExtParams maskCopyParams{1U, static_cast<uint32_t>(args.maskSlotSize), 0U, 0U, 0U};
    GlobalTensor<uint8_t> dstGlobalTensor;

    const MoeTopologyContext &topology = context.topology;
    int32_t dstRank = curExpertId / static_cast<int32_t>(topology.expertPerRank);
    int32_t expertNumPer = curExpertId % static_cast<int32_t>(topology.expertPerRank);
    TEventID eventId = kBufEvents[iter % static_cast<int32_t>(DOUBLE_BUFFER)];
    LocalTensor<uint8_t> maskBuf = GetSendMaskTensor(scratch, iter);
    LocalTensor<uint32_t> maskBufU32 = maskBuf.template ReinterpretCast<uint32_t>();
    LocalTensor<int32_t> maskBufI32 = maskBuf.template ReinterpretCast<int32_t>();

    WaitFlag<AscendC::HardEvent::MTE3_V>(eventId);
    CompareScalar(maskBuf, scratch.topkIdsTensor, curExpertId, AscendC::CMPMODE::EQ, args.compareCount);
    uint64_t sendCnt = 0;
    GatherMask(scratch.sendGatherOutTensor, scratch.topkIdsTensor, maskBufU32, true,
        static_cast<uint32_t>(sendTotalNum), {1, 1, 0, 0}, sendCnt);
    SyncFuncStatic<AscendC::HardEvent::V_S, SYNC_EVENT_ID2>();
    maskBufI32.SetValue(countWordIdx, static_cast<int32_t>(sendCnt));
    SyncFuncStatic<AscendC::HardEvent::S_MTE3, SYNC_EVENT_ID3>();

    uint64_t dstOffset = args.maskWinOffset +
        static_cast<uint64_t>(expertNumPer * static_cast<int32_t>(topology.worldSize) +
        static_cast<int32_t>(topology.rankId)) * static_cast<uint64_t>(args.maskSlotSize);
    dstGlobalTensor.SetGlobalBuffer((__gm__ uint8_t*)(args.winRankAddr[dstRank] + dstOffset));
    DataCopyPad(dstGlobalTensor, maskBuf, maskCopyParams);
    SetFlag<AscendC::HardEvent::MTE3_V>(eventId);
}

__aicore__ inline void SendMaskCalOneCore(const DispatchPrepareContext &context, const SendMaskArgs &args,
    SendMaskScratch &scratch)
{
    if constexpr (g_coreType == AIC) {
        return;
    }
    const AivThreadContext &thread = context.thread;
    if (thread.threadNum == 0U || thread.threadIdx >= thread.threadNum) {
        return;
    }

    uint64_t sendTotalNum =
        static_cast<uint64_t>(context.tokenShape.tokenNum) * static_cast<uint64_t>(context.tokenShape.topK);
    GlobalTensor<int32_t> srcGlobalTensor;
    srcGlobalTensor.SetGlobalBuffer((__gm__ int32_t*)args.expertIdxGmAddr);
    Duplicate<int32_t>(scratch.topkIdsTensor, 0, args.compareCount);
    SyncFuncStatic<AscendC::HardEvent::V_MTE2, SYNC_EVENT_ID1>();
    DataCopyExtParams loadParams{1U, static_cast<uint32_t>(sendTotalNum * sizeof(int32_t)), 0U, 0U, 0U};
    DataCopyPadExtParams<int32_t> loadPad{false, 0U, 0U, 0U};
    DataCopyPad(scratch.topkIdsTensor, srcGlobalTensor, loadParams, loadPad);
    SyncFuncStatic<AscendC::HardEvent::MTE2_V, SYNC_EVENT_ID1>();

    constexpr TEventID kBufEvents[DOUBLE_BUFFER] = {EVENT_ID0, EVENT_ID1};
    for (int32_t index = 0; index < static_cast<int32_t>(DOUBLE_BUFFER); ++index) {
        SetFlag<AscendC::HardEvent::MTE3_V>(kBufEvents[index]);
    }
    int32_t totalExperts =
        static_cast<int32_t>(context.topology.worldSize * context.topology.expertPerRank);
    int32_t iter = 0;
    for (int32_t curExpertId = static_cast<int32_t>(thread.threadIdx); curExpertId < totalExperts;
        curExpertId += static_cast<int32_t>(thread.threadNum), ++iter) {
        SendMaskCalOneExpert(context, args, scratch, sendTotalNum, curExpertId, iter);
    }
    for (int32_t index = 0; index < static_cast<int32_t>(DOUBLE_BUFFER); ++index) {
        WaitFlag<AscendC::HardEvent::MTE3_V>(kBufEvents[index]);
    }
}

struct QuantProcessArgs {
    GM_ADDR xGmAddr;
    GM_ADDR quantTokenScalePtr;
    uint32_t quantTokenAlignBytes;
    uint32_t quantScaleAlignBytes;
    uint32_t quantTokenScaleAlignBytes;
    uint32_t quantScaleNumAlignPerToken;
};

template <typename ActivationType>
struct QuantProcessScratch {
    LocalTensor<bfloat16_t> xInTensor0;
    LocalTensor<bfloat16_t> xInTensor1;
    LocalTensor<ActivationType> xOutTensor0;
    LocalTensor<ActivationType> xOutTensor1;
    LocalTensor<uint16_t> mxTempTensor;
};

// QuantProcessInRankOneCore：量化当前逻辑线程负责的本卡 token。
// 依赖项由 DispatchPrepareContext/QuantProcessArgs/QuantProcessScratch 显式传入，
// 不读取物理核 ID 或 MegaMoe 成员。
template <int32_t QuantMode, typename QuantOutType, typename ActivationType>
__aicore__ inline void QuantProcessInRankOneCore(const DispatchPrepareContext &context,
    const QuantProcessArgs &args, QuantProcessScratch<ActivationType> &scratch)
{
    if constexpr(g_coreType == AIC) {
        return;
    }
    const AivThreadContext &thread = context.thread;
    if (thread.threadNum == 0U || thread.threadIdx >= thread.threadNum) {
        return;
    }

    int32_t currentNum;
    int32_t currentOffset;
    TilingByThreadContext(
        context.tokenShape.tokenNum, currentNum, currentOffset, thread.threadIdx, thread.threadNum, 1);
    uint32_t hiddenDim = context.tokenShape.tokenHiddenDim;
    GlobalTensor<bfloat16_t> srcGlobalTensor;
    GlobalTensor<uint8_t> dstGlobalTensor;
    DataCopyParams xCopyInParams = {1U, static_cast<uint16_t>(hiddenDim * sizeof(bfloat16_t)), 0U, 0U};
    DataCopyPadParams xCopyInPadParams{true, 0, 0, 0};
    DataCopyExtParams xCopyOutParams = {1U,
        static_cast<uint32_t>(args.quantTokenAlignBytes + args.quantScaleAlignBytes), 0U, 0U, 0U};
    SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
    for (int32_t index = 0; index < currentNum; index++) {
        srcGlobalTensor.SetGlobalBuffer((__gm__ bfloat16_t*)(args.xGmAddr +
            (currentOffset + index) * hiddenDim * sizeof(bfloat16_t)));
        dstGlobalTensor.SetGlobalBuffer((__gm__ uint8_t*)(args.quantTokenScalePtr +
            (currentOffset + index) * args.quantTokenScaleAlignBytes));
        auto event = (index % DOUBLE_BUFFER == 0) ? EVENT_ID0 : EVENT_ID1;
        auto xInTensor = (index % DOUBLE_BUFFER == 0) ? scratch.xInTensor0 : scratch.xInTensor1;
        auto xOutTensor = (index % DOUBLE_BUFFER == 0) ? scratch.xOutTensor0 : scratch.xOutTensor1;
        WaitFlag<AscendC::HardEvent::MTE3_MTE2>(event);
        DataCopyPad(xInTensor, srcGlobalTensor, xCopyInParams, xCopyInPadParams);
        SetFlag<AscendC::HardEvent::MTE2_V>(event);
        WaitFlag<AscendC::HardEvent::MTE2_V>(event);
        __ubuf__ bfloat16_t* srcAddr = (__ubuf__ bfloat16_t*)xInTensor.GetPhyAddr();
        __ubuf__ uint16_t* maxExpAddr = (__ubuf__ uint16_t*)scratch.mxTempTensor.GetPhyAddr();
        __ubuf__ uint16_t* halfScaleAddr = (__ubuf__ uint16_t*)scratch.mxTempTensor[Ops::Base::CeilAlign(
            args.quantScaleNumAlignPerToken, static_cast<uint32_t>(ALIGN_32))].GetPhyAddr();
        __ubuf__ int8_t* outDataAddr = (__ubuf__ int8_t*)xOutTensor.GetPhyAddr();
        __ubuf__ uint16_t* mxScaleAddr = (__ubuf__ uint16_t*)xOutTensor[args.quantTokenAlignBytes].GetPhyAddr();

        Quant::ComputeMaxExp(srcAddr, maxExpAddr, hiddenDim);
        Quant::ComputeScale<QuantOutType>(maxExpAddr, mxScaleAddr, halfScaleAddr,
            args.quantScaleNumAlignPerToken);
        if constexpr (QuantMode == E2M1_QUANT) {
            Quant::ComputeFp4Data<bfloat16_t, QuantOutType, AscendC::RoundMode::CAST_TRUNC,
                AscendC::RoundMode::CAST_RINT>(srcAddr, halfScaleAddr, outDataAddr, hiddenDim);
        } else {
            Quant::ComputeFp8Data<bfloat16_t, QuantOutType, AscendC::RoundMode::CAST_TRUNC,
                AscendC::RoundMode::CAST_RINT>(srcAddr, halfScaleAddr, outDataAddr, hiddenDim);
        }
        SetFlag<AscendC::HardEvent::V_MTE3>(event);
        WaitFlag<AscendC::HardEvent::V_MTE3>(event);
        auto xOutBytesTensor = xOutTensor.template ReinterpretCast<uint8_t>();
        DataCopyPad(dstGlobalTensor, xOutBytesTensor, xCopyOutParams);
        SetFlag<AscendC::HardEvent::MTE3_MTE2>(event);
    }
    WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
}

using Gmm1ProblemShape = Shape<int64_t, int64_t, int64_t, int64_t>;
using Gmm1BlockOffset = Shape<int64_t, int64_t, int64_t, int64_t, int64_t,
    int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t>;

struct Gmm1ExpertLoopState {
    Gmm1ProblemShape problemShape;
    Gmm1BlockOffset baseOffset;
    uint32_t expertBeforeCnt;
};

// GMM1 scheduling and count-workspace indexing are deliberately separate.
// GMM2 may use another block count while reading the same count workspace.
struct DispatchGmm1SwigluContext {
    ThreadBlockContext threadBlock;          // Logical dispatch/GMM1 scheduling.
    BlockWorkspaceContext countWorkspace;   // Fixed count layout shared with GMM2.
    uint32_t worldSize;
    uint32_t expertPerRank;
    uint32_t tokenHiddenDim;
    uint32_t gmm1OutputDim;
    uint32_t topK;
    uint32_t blockNumPerRank;
    uint64_t maxOutputSize;
    uint64_t sendTotalNum;
    uint32_t maskAlignSize;
    uint32_t maskSlotSize;
    uint64_t quantWinOffset;
    uint32_t quantTokenAlignBytes;
    uint32_t quantScaleAlignBytes;
    int32_t dispatchFlagSlotsPerExpert;
    int32_t groupedMatmulMode;
};

template <typename BlockEpilogue>
struct DispatchGmm1SwigluArgs {
    const Params *gmmParams;  // Read-only low-level GMM data contract; no MegaMoe object.
    BlockEpilogue *epilogueOp;
    GM_ADDR *winRankAddr;
    GM_ADDR maskRecvPtr;
    GM_ADDR expertRevTokenNumsPtr;
    GM_ADDR tripleInfoPtr;
    GM_ADDR cumsumInfoPtr;
    GM_ADDR dispatchRevDataPtr;
    GM_ADDR dispatchRevScalePtr;
    GM_ADDR gmm1MmadResPtr;
    GM_ADDR weight1Ptr;
    GM_ADDR weight1ScalePtr;
    GM_ADDR swigluToGmm2FlagPtr;
    GM_ADDR dispatchToGmm1FlagPtr;
    GM_ADDR sendCntFlagPtr;
    uint32_t *startBlockIdx;
    int32_t *vecSetSyncCom;
};

constexpr int32_t GMM1_DISPATCH_BUFFER_NUM = 6;

template <typename ActivationType>
struct DispatchGmm1SwigluScratch {
    GlobalTensor<int32_t> expertRevNumsGlobalTensor;
    GlobalTensor<int32_t> cumsumInfoGlobalTensor;
    LocalTensor<int32_t> topkIndexTensor;
    LocalTensor<uint8_t> gatherMaskTensor;
    LocalTensor<uint32_t> gatherMaskInt32Tensor;
    LocalTensor<int32_t> expertTokenCntTensor;
    LocalTensor<int32_t> validTopkIndexTensor;
    LocalTensor<int32_t> cumsumInfoTensor;
    LocalTensor<ActivationType> copyTmpTensors[GMM1_DISPATCH_BUFFER_NUM];
    LocalTensor<int32_t> tripleTensor;
    LocalTensor<int32_t> expertTokenNumsOutTensor;
    uint64_t cumsumRevCntInRank;
};

template <typename ActivationType, bool EnableA8W4, typename BlockEpilogue>
__aicore__ inline void InitDispatchGmm1SwigluBuffers(
    const DispatchGmm1SwigluContext &context,
    const DispatchGmm1SwigluArgs<BlockEpilogue> &args,
    DispatchGmm1SwigluScratch<ActivationType> &scratch)
{
    scratch.expertRevNumsGlobalTensor.SetGlobalBuffer(
        reinterpret_cast<__gm__ int32_t *>(args.expertRevTokenNumsPtr));
    if constexpr (EnableA8W4) {
        uint64_t cumsumStride = Ops::Base::CeilAlign(
            static_cast<int64_t>(context.worldSize * context.expertPerRank * sizeof(int32_t)), ALIGN_32);
        scratch.cumsumInfoGlobalTensor.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(
            args.cumsumInfoPtr + cumsumStride * context.countWorkspace.blockIdx));
    }
    scratch.cumsumRevCntInRank = 0;

    if constexpr (g_coreType == AIC) {
        return;
    }

    uint32_t expertTokenCntTensorAddr = 0;
    uint32_t expertTokenCntTensorSize = ALIGN_32;
    scratch.expertTokenCntTensor = LocalTensor<int32_t>(TPosition::VECCALC,
        expertTokenCntTensorAddr, expertTokenCntTensorSize / sizeof(int32_t));

    uint32_t cumsumInfoTensorAddr = expertTokenCntTensorAddr + expertTokenCntTensorSize;
    uint32_t cumsumInfoTensorSize = Ops::Base::CeilAlign(
        static_cast<int64_t>(context.worldSize * context.expertPerRank * sizeof(int32_t)),
        static_cast<int64_t>(ALIGN_32));
    scratch.cumsumInfoTensor = LocalTensor<int32_t>(TPosition::VECCALC,
        cumsumInfoTensorAddr, cumsumInfoTensorSize / sizeof(int32_t));

    uint32_t gatherMaskTensorAddr = cumsumInfoTensorAddr + cumsumInfoTensorSize;
    uint32_t gatherMaskTensorSize = context.maskSlotSize * context.worldSize;
    scratch.gatherMaskTensor = LocalTensor<uint8_t>(TPosition::VECCALC,
        gatherMaskTensorAddr, gatherMaskTensorSize);
    scratch.gatherMaskInt32Tensor = LocalTensor<uint32_t>(TPosition::VECCALC,
        gatherMaskTensorAddr, gatherMaskTensorSize / sizeof(uint32_t));

    uint32_t validTopkIndexTensorAddr = gatherMaskTensorAddr + gatherMaskTensorSize;
    uint32_t validTopkIndexTensorSize = Ops::Base::CeilAlign(
        static_cast<int64_t>(context.sendTotalNum * sizeof(int32_t)), static_cast<int64_t>(ALIGN_32));
    scratch.validTopkIndexTensor = LocalTensor<int32_t>(TPosition::VECCALC,
        validTopkIndexTensorAddr, validTopkIndexTensorSize / sizeof(int32_t));

    uint32_t topkIndexTensorAddr = validTopkIndexTensorAddr + validTopkIndexTensorSize;
    uint32_t topkIndexTensorSize = validTopkIndexTensorSize;
    scratch.topkIndexTensor = LocalTensor<int32_t>(TPosition::VECCALC,
        topkIndexTensorAddr, topkIndexTensorSize / sizeof(int32_t));

    uint32_t tokenScaleSize = Ops::Base::CeilAlign(
        static_cast<int64_t>(context.quantTokenAlignBytes + context.quantScaleAlignBytes),
        static_cast<int64_t>(ALIGN_32));
    uint32_t copyTmpBaseAddr = topkIndexTensorAddr + topkIndexTensorSize;
    for (int32_t index = 0; index < GMM1_DISPATCH_BUFFER_NUM; ++index) {
        scratch.copyTmpTensors[index] = LocalTensor<ActivationType>(TPosition::VECCALC,
            copyTmpBaseAddr + static_cast<uint32_t>(index) * tokenScaleSize,
            tokenScaleSize / sizeof(ActivationType));
    }

    uint32_t expertTokenNumsOutTensorAddr = copyTmpBaseAddr +
        static_cast<uint32_t>(GMM1_DISPATCH_BUFFER_NUM) * tokenScaleSize;
    uint32_t expertTokenNumsOutTensorSize = Ops::Base::CeilAlign(
        static_cast<int64_t>(context.expertPerRank * sizeof(int32_t)), static_cast<int64_t>(ALIGN_32));
    scratch.expertTokenNumsOutTensor = LocalTensor<int32_t>(TPosition::VECCALC,
        expertTokenNumsOutTensorAddr, expertTokenNumsOutTensorSize / sizeof(int32_t));
    uint32_t tripleTensorAddr = expertTokenNumsOutTensorAddr + expertTokenNumsOutTensorSize;
    uint32_t tripleTensorSize = GMM1_DISPATCH_BUFFER_NUM * INT32_PER_256B * sizeof(int32_t);
    scratch.tripleTensor = LocalTensor<int32_t>(TPosition::VECCALC,
        tripleTensorAddr, tripleTensorSize / sizeof(int32_t));

    CreateVecIndex(scratch.topkIndexTensor, 0, topkIndexTensorSize / sizeof(int32_t));
    Duplicate<int32_t>(scratch.cumsumInfoTensor, 0, cumsumInfoTensorSize / sizeof(int32_t));
    PipeBarrier<PIPE_ALL>();
}

__aicore__ inline Gmm1ExpertLoopState CreateGmm1ExpertLoopState(
    const DispatchGmm1SwigluContext &context)
{
    Gmm1ProblemShape shape;
    Get<N_VALUE>(shape) = context.gmm1OutputDim;
    Get<K_VALUE>(shape) = context.tokenHiddenDim;
    Gmm1BlockOffset offset{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    return Gmm1ExpertLoopState{shape, offset, 0};
}

template <typename ActivationType, typename WeightType, typename SwigluOutType, typename BlockEpilogue>
__aicore__ inline bool UpdateGmm1GroupParams(
    const DispatchGmm1SwigluContext &context,
    const DispatchGmm1SwigluArgs<BlockEpilogue> &args,
    DispatchGmm1SwigluScratch<ActivationType> &scratch,
    Gmm1ExpertLoopState &state, uint32_t expertIdx, uint64_t sendCnt)
{
    constexpr uint32_t aElemsPerByte = Std::IsSame<ActivationType, uint8_t>::value ? 2U : 1U;
    constexpr uint32_t bElemsPerByte = Std::IsSame<WeightType, fp4x2_e2m1_t>::value ? 2U : 1U;
    constexpr uint32_t cElemsPerByte = Std::IsSame<SwigluOutType, fp4x2_e2m1_t>::value ? 2U : 1U;

    if (expertIdx != 0) {
        uint64_t m = Get<M_VALUE>(state.problemShape);
        uint64_t n = Get<N_VALUE>(state.problemShape);
        uint64_t k = Get<K_VALUE>(state.problemShape);
        state.expertBeforeCnt += m;
        Get<IDX_A_OFFSET>(state.baseOffset) += m * k / aElemsPerByte;
        Get<IDX_B_OFFSET>(state.baseOffset) += n * k / bElemsPerByte;
        auto scaleK = Ops::Base::CeilDiv(k, static_cast<uint64_t>(MXFP_DIVISOR_SIZE)) * MXFP_MULTI_BASE_SIZE;
        Get<IDX_A_SCALE_OFFSET>(state.baseOffset) += m * scaleK;
        Get<IDX_B_SCALE_OFFSET>(state.baseOffset) += n * scaleK;
        Get<IDX_C_OFFSET>(state.baseOffset) += m * n / SWIGLU_N_HALF / cElemsPerByte;
        Get<IDX_C_SCALE_OFFSET>(state.baseOffset) +=
            m * Ops::Base::CeilDiv(n / SWIGLU_N_HALF,
                static_cast<uint64_t>(MXFP_DIVISOR_SIZE)) * MXFP_MULTI_BASE_SIZE;
        Get<IDX_FLAG_OFFSET>(state.baseOffset) += 1;
        Get<IDX_B2_OFFSET>(state.baseOffset) += k * n / SWIGLU_N_HALF / bElemsPerByte;
        Get<IDX_B2_SCALE_OFFSET>(state.baseOffset) +=
            k * Ops::Base::CeilDiv(n / SWIGLU_N_HALF,
                static_cast<uint64_t>(MXFP_DIVISOR_SIZE)) * MXFP_MULTI_BASE_SIZE;
        Get<IDX_Y2_OFFSET>(state.baseOffset) += m * k;
        Get<IDX_M_OFFSET>(state.baseOffset) += m;
        Get<IDX_GMM1_OFFSET>(state.baseOffset) += m * n;
        Get<IDX_GMM2_OFFSET>(state.baseOffset) += m * k;
    }

    if (GetSubBlockIdx() == 0) {
        __gm__ int32_t *sendCntFlag = reinterpret_cast<__gm__ int32_t *>(
            args.sendCntFlagPtr) +
            static_cast<uint64_t>(expertIdx) * context.countWorkspace.blockNum * INT_CACHELINE +
            static_cast<uint64_t>(context.countWorkspace.blockIdx) * INT_CACHELINE;
        while (AscendC::ReadGmByPassDCache(sendCntFlag) == 0) {
            int64_t startCycle = AscendC::GetSystemCycle();
            while (AscendC::GetSystemCycle() - startCycle < 100) {
            }
        }
        uint64_t offset = expertIdx * INT32_PER_256B * context.countWorkspace.blockNum +
            INT32_PER_256B * context.countWorkspace.blockIdx;
        DataCacheCleanAndInvalid<int32_t, CacheLine::ENTIRE_DATA_CACHE, DcciDst::CACHELINE_OUT>(
            scratch.expertRevNumsGlobalTensor[offset]);
        Get<M_VALUE>(state.problemShape) = scratch.expertRevNumsGlobalTensor.GetValue(offset);
    } else {
        Get<M_VALUE>(state.problemShape) = sendCnt;
    }
    return Get<M_VALUE>(state.problemShape) != 0;
}

template <typename ActivationType, bool EnableA8W4, typename BlockEpilogue>
__aicore__ inline void SendCntCalOneCore(
    const DispatchGmm1SwigluContext &context,
    const DispatchGmm1SwigluArgs<BlockEpilogue> &args,
    DispatchGmm1SwigluScratch<ActivationType> &scratch,
    uint32_t localExpertId, uint64_t &sendCnt)
{
    sendCnt = 0;
    uint32_t slotWordStride = context.maskSlotSize / sizeof(uint32_t);
    uint32_t countWordIdx = context.maskAlignSize / sizeof(uint32_t);
    GlobalTensor<uint8_t> maskSrcGlobal;
    maskSrcGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(
        args.maskRecvPtr +
        static_cast<uint64_t>(localExpertId) * context.worldSize * context.maskSlotSize));
    DataCopy(scratch.gatherMaskTensor, maskSrcGlobal, context.worldSize * context.maskSlotSize);

    if constexpr (EnableA8W4) {
        if (localExpertId != 0) {
            DataCopyPad(scratch.cumsumInfoTensor, scratch.cumsumInfoGlobalTensor,
                {1U, static_cast<uint32_t>(context.worldSize * localExpertId * sizeof(int32_t)), 0U, 0U, 0U},
                {true, 0U, 0U, 0U});
        }
    }
    SyncFuncStatic<AscendC::HardEvent::MTE2_S, SYNC_EVENT_ID2>();
    SyncFuncStatic<AscendC::HardEvent::MTE2_V, SYNC_EVENT_ID1>();

    for (uint32_t rankIdx = 0; rankIdx < context.worldSize; ++rankIdx) {
        int32_t rankCount = scratch.gatherMaskInt32Tensor.GetValue(rankIdx * slotWordStride + countWordIdx);
        sendCnt += static_cast<uint64_t>(rankCount);
        scratch.cumsumRevCntInRank += static_cast<uint64_t>(rankCount);
        scratch.cumsumInfoTensor.SetValue(localExpertId * context.worldSize + rankIdx,
            static_cast<int32_t>(scratch.cumsumRevCntInRank));
    }

    scratch.expertTokenCntTensor.SetValue(0, sendCnt);
    SyncFuncStatic<AscendC::HardEvent::S_MTE3, SYNC_EVENT_ID2>();
    uint64_t countOffset = localExpertId * INT32_PER_256B * context.countWorkspace.blockNum +
        INT32_PER_256B * context.countWorkspace.blockIdx;
    DataCopy<int32_t>(scratch.expertRevNumsGlobalTensor[countOffset],
        scratch.expertTokenCntTensor, INT32_PER_256B);
    if constexpr (EnableA8W4) {
        DataCopyPad(scratch.cumsumInfoGlobalTensor, scratch.cumsumInfoTensor,
            {1U, static_cast<uint32_t>(context.worldSize * (localExpertId + 1) * sizeof(int32_t)), 0U, 0U, 0U});
    }
    PipeBarrier<PIPE_ALL>();

    __gm__ int32_t *sendCntFlag = reinterpret_cast<__gm__ int32_t *>(
        args.sendCntFlagPtr) +
        static_cast<uint64_t>(localExpertId) * context.countWorkspace.blockNum * INT_CACHELINE +
        static_cast<uint64_t>(context.countWorkspace.blockIdx) * INT_CACHELINE;
    AscendC::AtomicAdd(sendCntFlag, static_cast<int32_t>(1));
}

template <typename ActivationType, typename QuantScaleType, typename BlockEpilogue>
__aicore__ inline void CopyDispatchTokensOneCore(
    const DispatchGmm1SwigluContext &context,
    const DispatchGmm1SwigluArgs<BlockEpilogue> &args,
    DispatchGmm1SwigluScratch<ActivationType> &scratch,
    int32_t rowDstOffset, int32_t remoteRankIdx, int32_t copyStartIdx, int32_t copyNum)
{
    if (copyNum <= 0) {
        return;
    }
    constexpr TEventID events[GMM1_DISPATCH_BUFFER_NUM] = {
        EVENT_ID0, EVENT_ID1, EVENT_ID2, EVENT_ID3, EVENT_ID4, EVENT_ID5};
    constexpr uint32_t activationElemsPerByte = Std::IsSame<ActivationType, uint8_t>::value ? 2U : 1U;
    int64_t widthA = context.tokenHiddenDim / activationElemsPerByte;
    int64_t widthAScale = Ops::Base::CeilDiv(
        static_cast<int64_t>(context.tokenHiddenDim), static_cast<int64_t>(MXFP_DIVISOR_SIZE)) *
        MXFP_MULTI_BASE_SIZE;
    uint32_t copyInNum = Ops::Base::CeilAlign(
        static_cast<int64_t>(context.quantTokenAlignBytes + context.quantScaleAlignBytes),
        static_cast<int64_t>(ALIGN_32));

    GlobalTensor<ActivationType> remoteRankGlobalTensor;
    GlobalTensor<ActivationType> tokenRevGlobalTensor;
    GlobalTensor<QuantScaleType> scaleRevGlobalTensor;
    GlobalTensor<int32_t> tripleGlobalTensor;
    tokenRevGlobalTensor.SetGlobalBuffer(reinterpret_cast<__gm__ ActivationType *>(
        args.dispatchRevDataPtr + rowDstOffset * widthA));
    scaleRevGlobalTensor.SetGlobalBuffer(reinterpret_cast<__gm__ QuantScaleType *>(
        args.dispatchRevScalePtr + rowDstOffset * widthAScale));
    remoteRankGlobalTensor.SetGlobalBuffer(reinterpret_cast<__gm__ ActivationType *>(
        args.winRankAddr[remoteRankIdx] + context.quantWinOffset));
    tripleGlobalTensor.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(
        args.tripleInfoPtr + rowDstOffset * INT32_PER_256B * sizeof(int32_t)));

    PipeBarrier<PIPE_ALL>();
    int32_t primeCount = copyNum < GMM1_DISPATCH_BUFFER_NUM ? copyNum : GMM1_DISPATCH_BUFFER_NUM;
    for (int32_t primeIdx = 0; primeIdx < primeCount; ++primeIdx) {
        int32_t topkIndex = scratch.validTopkIndexTensor.GetValue(copyStartIdx + primeIdx);
        int32_t tokenIndex = topkIndex / static_cast<int32_t>(context.topK);
        uint64_t remoteOffset = static_cast<uint64_t>(tokenIndex) * copyInNum;
        TEventID eventId = events[primeIdx];
        DataCopy(scratch.copyTmpTensors[primeIdx], remoteRankGlobalTensor[remoteOffset], copyInNum);
        SetFlag<AscendC::HardEvent::MTE2_MTE3>(eventId);
        scratch.tripleTensor[primeIdx * INT32_PER_256B].SetValue(RANK_ID, remoteRankIdx);
        scratch.tripleTensor[primeIdx * INT32_PER_256B].SetValue(TOKEN_ID, tokenIndex);
        scratch.tripleTensor[primeIdx * INT32_PER_256B].SetValue(TOPK_INDEX,
            topkIndex % static_cast<int32_t>(context.topK));
        SetFlag<AscendC::HardEvent::S_MTE3>(eventId);
    }

    for (int32_t copyIdx = 0; copyIdx < copyNum; ++copyIdx) {
        int32_t bufferIdx = copyIdx % GMM1_DISPATCH_BUFFER_NUM;
        TEventID eventId = events[bufferIdx];
        WaitFlag<AscendC::HardEvent::MTE2_MTE3>(eventId);
        LocalTensor<ActivationType> tokenScaleBuffer = scratch.copyTmpTensors[bufferIdx];
        LocalTensor<QuantScaleType> scaleBuffer = tokenScaleBuffer[
            context.quantTokenAlignBytes].template ReinterpretCast<QuantScaleType>();
        DataCopyPad(tokenRevGlobalTensor[copyIdx * widthA], tokenScaleBuffer,
            {1, static_cast<uint16_t>(widthA * sizeof(ActivationType)), 0U, 0U, 0U});
        DataCopyPad(scaleRevGlobalTensor[copyIdx * widthAScale], scaleBuffer,
            {1, static_cast<uint16_t>(widthAScale * sizeof(QuantScaleType)), 0U, 0U, 0U});
        WaitFlag<AscendC::HardEvent::S_MTE3>(eventId);
        DataCopy(tripleGlobalTensor[copyIdx * INT32_PER_256B],
            scratch.tripleTensor[bufferIdx * INT32_PER_256B], INT32_PER_256B);
        SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventId);
        SetFlag<AscendC::HardEvent::MTE3_S>(eventId);

        int32_t nextIdx = copyIdx + GMM1_DISPATCH_BUFFER_NUM;
        if (nextIdx < copyNum) {
            int32_t nextTopkIndex = scratch.validTopkIndexTensor.GetValue(copyStartIdx + nextIdx);
            int32_t nextTokenIndex = nextTopkIndex / static_cast<int32_t>(context.topK);
            uint64_t remoteOffset = static_cast<uint64_t>(nextTokenIndex) * copyInNum;
            WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventId);
            DataCopy(scratch.copyTmpTensors[bufferIdx], remoteRankGlobalTensor[remoteOffset], copyInNum);
            SetFlag<AscendC::HardEvent::MTE2_MTE3>(eventId);
            WaitFlag<AscendC::HardEvent::MTE3_S>(eventId);
            scratch.tripleTensor[bufferIdx * INT32_PER_256B].SetValue(RANK_ID, remoteRankIdx);
            scratch.tripleTensor[bufferIdx * INT32_PER_256B].SetValue(TOKEN_ID, nextTokenIndex);
            scratch.tripleTensor[bufferIdx * INT32_PER_256B].SetValue(TOPK_INDEX,
                nextTopkIndex % static_cast<int32_t>(context.topK));
            SetFlag<AscendC::HardEvent::S_MTE3>(eventId);
        }
    }
    for (int32_t bufferIdx = 0; bufferIdx < primeCount; ++bufferIdx) {
        WaitFlag<AscendC::HardEvent::MTE3_MTE2>(events[bufferIdx]);
        WaitFlag<AscendC::HardEvent::MTE3_S>(events[bufferIdx]);
    }
}

template <typename ActivationType, typename QuantScaleType, bool EnableA8W4, typename BlockEpilogue>
__aicore__ inline void DispatchTokensOneCore(
    const DispatchGmm1SwigluContext &context,
    const DispatchGmm1SwigluArgs<BlockEpilogue> &args,
    DispatchGmm1SwigluScratch<ActivationType> &scratch,
    GMMAddrInfo &gmmAddrInfo, uint32_t localExpertId)
{
    constexpr int32_t tileM = static_cast<int32_t>(MegaMoeImpl::L1_TILE_M_256);
    int32_t priorExpertCumsum = localExpertId == 0 ? 0 :
        scratch.cumsumInfoTensor.GetValue(localExpertId * context.worldSize - 1);
    uint32_t topkIndexTensorSize = Ops::Base::CeilAlign(
        static_cast<int64_t>(context.sendTotalNum * sizeof(int32_t)), static_cast<int64_t>(ALIGN_32));
    if constexpr (EnableA8W4) {
        if (localExpertId != 0) {
            CreateVecIndex(scratch.topkIndexTensor, 0, topkIndexTensorSize / sizeof(int32_t));
            PipeBarrier<PIPE_V>();
        }
    }

    for (uint32_t index = context.threadBlock.threadBlockIdx;
        index < context.worldSize * context.blockNumPerRank;
        index += context.threadBlock.threadBlockNum) {
        uint32_t dstRankIdx = index / context.blockNumPerRank;
        uint64_t tokenCount = 0;
        GatherMask(scratch.validTopkIndexTensor, scratch.topkIndexTensor,
            scratch.gatherMaskInt32Tensor[dstRankIdx * (context.maskSlotSize / sizeof(uint32_t))],
            true, context.sendTotalNum, {1, 1, 0, 0}, tokenCount);
        uint32_t aivIdxInGroup = index % context.blockNumPerRank;
        int32_t rowStart = (dstRankIdx == 0 && localExpertId == 0) ? 0 :
            scratch.cumsumInfoTensor.GetValue(localExpertId * context.worldSize + dstRankIdx - 1);
        int32_t rowsThisCore = 0;
        int32_t rowDstOffset = 0;
        if (rowStart < static_cast<int32_t>(context.maxOutputSize)) {
            int32_t rowsCopyCount = static_cast<int32_t>(tokenCount);
            if (rowStart + rowsCopyCount > static_cast<int32_t>(context.maxOutputSize)) {
                rowsCopyCount = static_cast<int32_t>(context.maxOutputSize) - rowStart;
            }
            int32_t rowsPerCore = Ops::Base::CeilDiv(
                rowsCopyCount, static_cast<int32_t>(context.blockNumPerRank));
            int32_t rowSrcOffset = static_cast<int32_t>(aivIdxInGroup) * rowsPerCore;
            rowDstOffset = rowStart + rowSrcOffset;
            if (rowDstOffset + rowsPerCore > rowStart + rowsCopyCount) {
                rowsPerCore = rowStart + rowsCopyCount - rowDstOffset;
            }
            if (rowsPerCore > 0) {
                SyncFuncStatic<AscendC::HardEvent::V_S, SYNC_EVENT_ID4>();
                CopyDispatchTokensOneCore<ActivationType, QuantScaleType>(
                    context, args, scratch, rowDstOffset, dstRankIdx, rowSrcOffset, rowsPerCore);
                rowsThisCore = rowsPerCore;
            }
        }

        if (rowsThisCore > 0) {
            SyncFuncStatic<AscendC::HardEvent::MTE3_S, SYNC_EVENT_ID5>();
            int32_t rowStartLocal = rowDstOffset - priorExpertCumsum;
            int32_t rowEndLocal = rowStartLocal + rowsThisCore;
            int32_t waveLo = rowStartLocal / tileM;
            int32_t waveHi = (rowEndLocal - 1) / tileM;
            for (int32_t wave = waveLo; wave <= waveHi; ++wave) {
                int32_t waveStart = wave * tileM;
                int32_t waveEnd = waveStart + tileM;
                int32_t lo = rowStartLocal > waveStart ? rowStartLocal : waveStart;
                int32_t hi = rowEndLocal < waveEnd ? rowEndLocal : waveEnd;
                AtomicAdd(gmmAddrInfo.dispatchToGmm1Flag + wave, static_cast<int32_t>(hi - lo));
            }
        }
    }
}

template <typename ActivationType, typename WeightType, typename SwigluOutType,
    typename QuantScaleType, bool EnableA8W4, typename BlockEpilogue>
__aicore__ inline void UpdateGmm1GlobalBuffer(
    const DispatchGmm1SwigluContext &context,
    const DispatchGmm1SwigluArgs<BlockEpilogue> &args,
    GMMAddrInfo &gmmAddrInfo, const Gmm1ExpertLoopState &state)
{
    if constexpr (EnableA8W4) {
        gmmAddrInfo.gmm1OutGlobal = args.gmm1MmadResPtr +
            Get<IDX_GMM1_OFFSET>(state.baseOffset) * sizeof(bfloat16_t);
    }
    gmmAddrInfo.aGlobal = args.dispatchRevDataPtr +
        Get<IDX_A_OFFSET>(state.baseOffset) * sizeof(ActivationType);
    gmmAddrInfo.aScaleGlobal = args.dispatchRevScalePtr +
        Get<IDX_A_SCALE_OFFSET>(state.baseOffset) * sizeof(QuantScaleType);
    gmmAddrInfo.bGlobal = args.weight1Ptr +
        Get<IDX_B_OFFSET>(state.baseOffset) * sizeof(ActivationType);
    gmmAddrInfo.bScaleGlobal = args.weight1ScalePtr +
        Get<IDX_B_SCALE_OFFSET>(state.baseOffset) * sizeof(QuantScaleType);
    if constexpr (g_coreType == AIV) {
        AscendC::Coord<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t> vecBaseOffset{
            Get<IDX_C_OFFSET>(state.baseOffset), Get<IDX_C_SCALE_OFFSET>(state.baseOffset),
            Get<IDX_FLAG_OFFSET>(state.baseOffset), 0L, 0L, 0L};
        args.epilogueOp->UpdateGlobalAddr(vecBaseOffset);
    }
    gmmAddrInfo.swigluToGmm2Flag = reinterpret_cast<__gm__ int32_t *>(
        args.swigluToGmm2FlagPtr) +
        Get<IDX_FLAG_OFFSET>(state.baseOffset) * INT_CACHELINE;
    gmmAddrInfo.dispatchToGmm1Flag = reinterpret_cast<__gm__ int32_t *>(
        args.dispatchToGmm1FlagPtr) +
        Get<IDX_FLAG_OFFSET>(state.baseOffset) * context.dispatchFlagSlotsPerExpert;
}

template <typename QuantOutType, typename ActivationType, typename WeightType,
    typename SwigluOutType, typename QuantScaleType, bool EnableA8W4, typename BlockEpilogue>
__aicore__ inline void RunGmm1SwigluStageOneCore(
    const DispatchGmm1SwigluContext &context,
    const DispatchGmm1SwigluArgs<BlockEpilogue> &args,
    const GMMAddrInfo &gmmAddrInfo, const Gmm1ExpertLoopState &state)
{
    if constexpr (EnableA8W4) {
        MegaMoeImpl::GroupMatmulSwigluQuantA8W4<
            QuantOutType, WeightType, bfloat16_t, QuantScaleType, QuantScaleType>(
            *args.epilogueOp, *args.gmmParams, state.problemShape, gmmAddrInfo,
            *args.startBlockIdx, *args.vecSetSyncCom, context.threadBlock);
    } else if (context.groupedMatmulMode == GROUPED_MATMUL_MODE_A8W8_NZ ||
        context.groupedMatmulMode == GROUPED_MATMUL_MODE_A4W4_NZ) {
        MegaMoeImpl::GroupMatmulSwigluQuant<
            QuantOutType, SwigluOutType, QuantOutType, bfloat16_t, QuantScaleType, QuantScaleType, true>(
            *args.epilogueOp, *args.gmmParams, state.problemShape, gmmAddrInfo,
            *args.startBlockIdx, *args.vecSetSyncCom, context.threadBlock);
    } else {
        MegaMoeImpl::GroupMatmulSwigluQuant<
            QuantOutType, SwigluOutType, QuantOutType, bfloat16_t, QuantScaleType, QuantScaleType>(
            *args.epilogueOp, *args.gmmParams, state.problemShape, gmmAddrInfo,
            *args.startBlockIdx, *args.vecSetSyncCom, context.threadBlock);
    }
}

// Every physical core in one MIX_AIC_1_2 block calls this function with the
// same logical threadBlock and countWorkspace contexts. The function preserves
// dispatch(e + 1) prefetch and GMM1/Swiglu(e) consumption order.
template <typename QuantOutType, typename ActivationType, typename WeightType,
    typename SwigluOutType, typename QuantScaleType, bool EnableA8W4, typename BlockEpilogue>
__aicore__ inline void DispatchGmm1SwigluOneCore(
    const DispatchGmm1SwigluContext &context,
    const DispatchGmm1SwigluArgs<BlockEpilogue> &args,
    DispatchGmm1SwigluScratch<ActivationType> &scratch)
{
    if (context.threadBlock.threadBlockNum == 0U ||
        context.threadBlock.threadBlockIdx >= context.threadBlock.threadBlockNum ||
        context.countWorkspace.blockNum == 0U ||
        context.countWorkspace.blockIdx >= context.countWorkspace.blockNum ||
        context.expertPerRank == 0U || context.blockNumPerRank == 0U) {
        return;
    }
    InitDispatchGmm1SwigluBuffers<ActivationType, EnableA8W4>(context, args, scratch);
    Gmm1ExpertLoopState dispatchState = CreateGmm1ExpertLoopState(context);
    Gmm1ExpertLoopState gmm1State = CreateGmm1ExpertLoopState(context);
    GMMAddrInfo dispatchAddrInfo;
    GMMAddrInfo gmm1AddrInfo;
    uint64_t currentSendCnt = 0;
    uint64_t nextSendCnt = 0;

    if constexpr (g_coreType == AIV) {
        if (GetSubBlockIdx() == 1) {
            SendCntCalOneCore<ActivationType, EnableA8W4>(context, args, scratch, 0, nextSendCnt);
            if (UpdateGmm1GroupParams<ActivationType, WeightType, SwigluOutType>(
                context, args, scratch, dispatchState, 0, nextSendCnt)) {
                UpdateGmm1GlobalBuffer<ActivationType, WeightType, SwigluOutType,
                    QuantScaleType, EnableA8W4>(context, args, dispatchAddrInfo, dispatchState);
                DispatchTokensOneCore<ActivationType, QuantScaleType, EnableA8W4>(
                    context, args, scratch, dispatchAddrInfo, 0);
            }
        }
    }

    for (uint32_t expertIdx = 0; expertIdx < context.expertPerRank; ++expertIdx) {
        currentSendCnt = nextSendCnt;
        if constexpr (g_coreType == AIV) {
            if (GetSubBlockIdx() == 1 && expertIdx + 1 < context.expertPerRank) {
                SendCntCalOneCore<ActivationType, EnableA8W4>(
                    context, args, scratch, expertIdx + 1, nextSendCnt);
                if (UpdateGmm1GroupParams<ActivationType, WeightType, SwigluOutType>(
                    context, args, scratch, dispatchState, expertIdx + 1, nextSendCnt)) {
                    UpdateGmm1GlobalBuffer<ActivationType, WeightType, SwigluOutType,
                        QuantScaleType, EnableA8W4>(context, args, dispatchAddrInfo, dispatchState);
                    DispatchTokensOneCore<ActivationType, QuantScaleType, EnableA8W4>(
                        context, args, scratch, dispatchAddrInfo, expertIdx + 1);
                }
            }
        }

        if (!UpdateGmm1GroupParams<ActivationType, WeightType, SwigluOutType>(
            context, args, scratch, gmm1State, expertIdx, currentSendCnt)) {
            continue;
        }
        UpdateGmm1GlobalBuffer<ActivationType, WeightType, SwigluOutType,
            QuantScaleType, EnableA8W4>(context, args, gmm1AddrInfo, gmm1State);
        RunGmm1SwigluStageOneCore<QuantOutType, ActivationType, WeightType,
            SwigluOutType, QuantScaleType, EnableA8W4>(
                context, args, gmm1AddrInfo, gmm1State);
    }
    EndSync(*args.vecSetSyncCom);
}

} // namespace MegaMoeImpl

#endif
