/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MEGA_MOE_AIV_DISPATCH_H
#define MEGA_MOE_AIV_DISPATCH_H

#include "../single_core_context.h"
#include "../mega_moe_impl.h"

namespace MegaMoeImpl {

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
    ThreadBlockContext threadBlock;
    BlockWorkspaceContext countWorkspace;
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
    const Params *gmmParams;
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

} // namespace MegaMoeImpl

#endif // MEGA_MOE_AIV_DISPATCH_H
