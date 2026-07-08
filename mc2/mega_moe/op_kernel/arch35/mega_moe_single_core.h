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
#if __has_include("../../moe_distribute_dispatch_v2/quantize_functions.h")
#include "../../moe_distribute_dispatch_v2/quantize_functions.h"
#else
#include "../../../moe_distribute_dispatch_v2/op_kernel/quantize_functions.h"
#endif

namespace MegaMoeImpl {

struct CoreContext {
    uint32_t blockIdx;
    uint32_t aivCoreIdx;
    uint32_t subBlockIdx;
    uint32_t blockNum;
    uint32_t blockAivNum;
};

__aicore__ inline void TilingByCoreContext(int32_t totalLen, int32_t &coreLen, int32_t &coreOffset,
    uint32_t coreIdx, uint32_t coreNum, int32_t align = ALIGN_32)
{
    int32_t lenPerCore = Ops::Base::CeilDiv(static_cast<uint32_t>(totalLen), coreNum);
    int32_t lenPerCoreAlign = Ops::Base::CeilAlign(static_cast<uint32_t>(lenPerCore), static_cast<uint32_t>(align));
    coreLen = lenPerCoreAlign;
    coreOffset = static_cast<int32_t>(coreIdx) * lenPerCoreAlign;
    if (coreOffset + coreLen >= totalLen) {
        coreLen = totalLen - coreOffset;
    }
    if (coreOffset >= totalLen) {
        coreLen = 0;
    }
}

struct ResetWorkspaceArgs {
    GM_ADDR flagSwiGluToGmm2Ptr;
    GM_ADDR gmm2CombineSyncCounterPtr;
    uint32_t expertPerRank;
    int32_t dispatchFlagSlotsPerExpert;
    uint32_t aicNum;
    bool resetCombineCounters;
};

__aicore__ inline void ResetGmm2CombineSyncCountersOneCore(const CoreContext &core,
    const ResetWorkspaceArgs &args, LocalTensor<int32_t> resetTensor)
{
    if constexpr (g_coreType == AIV) {
        int32_t totalCounters = static_cast<int32_t>(
            static_cast<int64_t>(args.expertPerRank) * core.blockAivNum * INT_CACHELINE);
        int32_t coreLen, coreOffset;
        TilingByCoreContext(totalCounters, coreLen, coreOffset, core.aivCoreIdx, core.blockAivNum);
        GlobalTensor<int32_t> gmm2CombineSyncCounterGm;
        gmm2CombineSyncCounterGm.SetGlobalBuffer((__gm__ int32_t*)args.gmm2CombineSyncCounterPtr);
        if (coreLen > 0) {
            Duplicate(resetTensor, 0, coreLen);
            SyncFuncStatic<AscendC::HardEvent::V_MTE3, SYNC_EVENT_ID2>();
            DataCopy(gmm2CombineSyncCounterGm[coreOffset], resetTensor, coreLen);
        }
    }
}

__aicore__ inline void ResetFlagListOneCore(const CoreContext &core, const ResetWorkspaceArgs &args,
    LocalTensor<int32_t> resetTensor)
{
    if constexpr (g_coreType == AIC) {
        return;
    }
    GlobalTensor<int32_t> swigluToGmm2FlagGm;
    swigluToGmm2FlagGm.SetGlobalBuffer((__gm__ int32_t*)args.flagSwiGluToGmm2Ptr);
    int32_t flagNum = static_cast<int32_t>(args.expertPerRank) *
        (static_cast<int32_t>(INT_CACHELINE) + args.dispatchFlagSlotsPerExpert +
        static_cast<int32_t>(INT_CACHELINE) * static_cast<int32_t>(args.aicNum));
    int32_t coreLen, coreOffset;
    TilingByCoreContext(flagNum, coreLen, coreOffset, core.aivCoreIdx, core.blockAivNum, 1);
    DataCopyExtParams rankSyncCopyParams{1U, static_cast<uint32_t>(coreLen * sizeof(int32_t)), 0U, 0U, 0U};
    SyncFuncStatic<AscendC::HardEvent::V_MTE3, SYNC_EVENT_ID2>();
    if (coreLen != 0) {
        DataCopyPad(swigluToGmm2FlagGm[coreOffset], resetTensor, rankSyncCopyParams);
    }
    if (args.resetCombineCounters) {
        ResetGmm2CombineSyncCountersOneCore(core, args, resetTensor);
    }
}

struct SendMaskArgs {
    GM_ADDR expertIdxGmAddr;
    GM_ADDR *winRankAddr;
    uint32_t rankId;
    uint32_t worldSize;
    uint32_t expertPerRank;
    uint64_t sendTotalNum;
    int32_t compareCount;
    uint32_t maskAlignSize;
    uint32_t maskSlotSize;
    uint64_t maskWinOffset;
};

struct SendMaskScratch {
    LocalTensor<int32_t> topkIdsTensor;
    LocalTensor<uint8_t> sendMaskTensor0;
    LocalTensor<uint8_t> sendMaskTensor1;
    LocalTensor<int32_t> sendGatherOutTensor;
};

__aicore__ inline LocalTensor<uint8_t> GetSendMaskTensor(SendMaskScratch &scratch, int32_t iter)
{
    return (iter % static_cast<int32_t>(DOUBLE_BUFFER) == 0) ? scratch.sendMaskTensor0 : scratch.sendMaskTensor1;
}

__aicore__ inline void SendMaskCalOneExpert(const SendMaskArgs &args, SendMaskScratch &scratch,
    int32_t curExpertId, int32_t iter)
{
    if constexpr (g_coreType == AIC) {
        return;
    }

    constexpr TEventID kBufEvents[DOUBLE_BUFFER] = {EVENT_ID0, EVENT_ID1};
    uint32_t countWordIdx = static_cast<uint32_t>(args.maskAlignSize) / sizeof(int32_t);
    DataCopyExtParams maskCopyParams{1U, static_cast<uint32_t>(args.maskSlotSize), 0U, 0U, 0U};
    GlobalTensor<uint8_t> dstGlobalTensor;

    int32_t dstRank = curExpertId / static_cast<int32_t>(args.expertPerRank);
    int32_t expertNumPer = curExpertId % static_cast<int32_t>(args.expertPerRank);
    TEventID eventId = kBufEvents[iter % static_cast<int32_t>(DOUBLE_BUFFER)];
    LocalTensor<uint8_t> maskBuf = GetSendMaskTensor(scratch, iter);
    LocalTensor<uint32_t> maskBufU32 = maskBuf.template ReinterpretCast<uint32_t>();
    LocalTensor<int32_t> maskBufI32 = maskBuf.template ReinterpretCast<int32_t>();

    WaitFlag<AscendC::HardEvent::MTE3_V>(eventId);
    CompareScalar(maskBuf, scratch.topkIdsTensor, curExpertId, AscendC::CMPMODE::EQ, args.compareCount);
    uint64_t sendCnt = 0;
    GatherMask(scratch.sendGatherOutTensor, scratch.topkIdsTensor, maskBufU32, true,
        static_cast<uint32_t>(args.sendTotalNum), {1, 1, 0, 0}, sendCnt);
    SyncFuncStatic<AscendC::HardEvent::V_S, SYNC_EVENT_ID2>();
    maskBufI32.SetValue(countWordIdx, static_cast<int32_t>(sendCnt));
    SyncFuncStatic<AscendC::HardEvent::S_MTE3, SYNC_EVENT_ID3>();

    uint64_t dstOffset = args.maskWinOffset +
        static_cast<uint64_t>(expertNumPer * static_cast<int32_t>(args.worldSize) +
        static_cast<int32_t>(args.rankId)) * static_cast<uint64_t>(args.maskSlotSize);
    dstGlobalTensor.SetGlobalBuffer((__gm__ uint8_t*)(args.winRankAddr[dstRank] + dstOffset));
    DataCopyPad(dstGlobalTensor, maskBuf, maskCopyParams);
    SetFlag<AscendC::HardEvent::MTE3_V>(eventId);
}

__aicore__ inline void SendMaskCalOneCore(const CoreContext &core, const SendMaskArgs &args,
    SendMaskScratch &scratch)
{
    if constexpr (g_coreType == AIC) {
        return;
    }

    GlobalTensor<int32_t> srcGlobalTensor;
    srcGlobalTensor.SetGlobalBuffer((__gm__ int32_t*)args.expertIdxGmAddr);
    Duplicate<int32_t>(scratch.topkIdsTensor, 0, args.compareCount);
    SyncFuncStatic<AscendC::HardEvent::V_MTE2, SYNC_EVENT_ID1>();
    DataCopyExtParams loadParams{1U, static_cast<uint32_t>(args.sendTotalNum * sizeof(int32_t)), 0U, 0U, 0U};
    DataCopyPadExtParams<int32_t> loadPad{false, 0U, 0U, 0U};
    DataCopyPad(scratch.topkIdsTensor, srcGlobalTensor, loadParams, loadPad);
    SyncFuncStatic<AscendC::HardEvent::MTE2_V, SYNC_EVENT_ID1>();

    constexpr TEventID kBufEvents[DOUBLE_BUFFER] = {EVENT_ID0, EVENT_ID1};
    for (int32_t index = 0; index < static_cast<int32_t>(DOUBLE_BUFFER); ++index) {
        SetFlag<AscendC::HardEvent::MTE3_V>(kBufEvents[index]);
    }
    int32_t totalExperts = static_cast<int32_t>(args.worldSize * args.expertPerRank);
    int32_t iter = 0;
    for (int32_t curExpertId = static_cast<int32_t>(core.aivCoreIdx); curExpertId < totalExperts;
        curExpertId += static_cast<int32_t>(core.blockAivNum), ++iter) {
        SendMaskCalOneExpert(args, scratch, curExpertId, iter);
    }
    for (int32_t index = 0; index < static_cast<int32_t>(DOUBLE_BUFFER); ++index) {
        WaitFlag<AscendC::HardEvent::MTE3_V>(kBufEvents[index]);
    }
}

struct QuantProcessArgs {
    GM_ADDR xGmAddr;
    GM_ADDR quantTokenScalePtr;
    int32_t tokenNum;
    uint32_t hiddenDim;
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

// QuantProcessInRankOneCore：当前核负责的本卡 token 量化。
// 依赖项由 CoreContext/QuantProcessArgs/QuantProcessScratch 显式传入，不读取 MegaMoe 成员。
template <int32_t QuantMode, typename QuantOutType, typename ActivationType>
__aicore__ inline void QuantProcessInRankOneCore(const CoreContext &core, const QuantProcessArgs &args,
    QuantProcessScratch<ActivationType> &scratch)
{
    if constexpr(g_coreType == AIC) {
        return;
    }

    int32_t currentNum;
    int32_t currentOffset;
    TilingByCoreContext(args.tokenNum, currentNum, currentOffset, core.aivCoreIdx, core.blockAivNum, 1);
    uint32_t hiddenDim = args.hiddenDim;
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

} // namespace MegaMoeImpl

#endif
