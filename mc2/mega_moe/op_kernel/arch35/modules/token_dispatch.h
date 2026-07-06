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
 * \file token_dispatch.h
 * \brief Single-core TokenDispatch module: quantize local tokens, compute triple info,
 *        dispatch tokens from remote cards, output expert token counts.
 */

#ifndef MEGA_MOE_TOKEN_DISPATCH_H
#define MEGA_MOE_TOKEN_DISPATCH_H

#include "kernel_operator.h"
#include "mega_moe_tile_info.h"
#include "mega_moe_base.h"
#include "mega_moe_workspace_info.h"
#if __has_include("../../common/mc2_kernel_utils.h")
#include "../../common/mc2_kernel_utils.h"
#else
#include "../../../common/op_kernel/mc2_kernel_utils.h"
#endif
#if __has_include("../../moe_distribute_dispatch_v2/quantize_functions.h")
#include "../../moe_distribute_dispatch_v2/quantize_functions.h"
#else
#include "../../../moe_distribute_dispatch_v2/op_kernel/quantize_functions.h"
#endif

using namespace AscendC;

namespace MegaMoeModules {
namespace TokenDispatch {

template <typename ActivationType, typename QuantOutType, int32_t QuantMode>
struct TokenDispatchState {
    LocalTensor<bfloat16_t> xInTensor1_;
    LocalTensor<bfloat16_t> xInTensor2_;
    LocalTensor<ActivationType> xOutTensor1_;
    LocalTensor<ActivationType> xOutTensor2_;
    LocalTensor<uint16_t> mxTempTensor_;

    LocalTensor<int32_t> expertTokenCntTensor_;
    LocalTensor<int32_t> cumsumInfoTensor_;
    LocalTensor<uint8_t> gatherMaskTensor_;
    LocalTensor<uint32_t> gatherMaskInt32Tensor_;
    LocalTensor<int32_t> validTopkIndexTensor_;
    LocalTensor<int32_t> topkIndexTensor_;
    LocalTensor<ActivationType> copyTmpTensors_[DISPATCH_BUFFER_NUM];
    LocalTensor<int32_t> tripleTensor_;
    LocalTensor<int32_t> expertTokenNumsOutTensor_;

    uint32_t ubBufferUsedAddr;
    uint32_t k;
    uint32_t topK;
    uint32_t m;
    uint32_t sendTotalNum;
    uint32_t worldSize;
    uint32_t expertPerRank;
    uint32_t rankId;
    uint32_t blockNumPerRank;
    uint64_t maxOutputSize;
    uint32_t mxQuantScaleNumAlignPerToken;
    uint32_t mxQuantTokenAlignBytes;
    uint32_t mxQuantScaleAlignBytes;
    uint32_t mxQuantTokenScaleAlignBytes;
    uint64_t quantWinOffset;
    uint64_t cumsumRevCntInRank;
    int32_t compareCount;
    uint32_t maskAlignSize;
    uint32_t maskSlotSize;

    static constexpr uint32_t A_ELEMS_PER_BYTE = Std::IsSame<QuantOutType, fp4x2_e2m1_t>::value ? 2U : 1U;

    Params params;
};

template <typename ActivationType, typename QuantOutType, int32_t QuantMode>
__aicore__ inline void QuantProcessInRank(
    TokenDispatchState<ActivationType, QuantOutType, QuantMode> &state, const TileInfo &tile)
{
    if constexpr(g_coreType == AIC) {
        return;
    }

    int32_t currentNum;
    int32_t currentOffset;
    TilingByTile(state.m, currentNum, currentOffset, tile.aivTileIdx, tile.aivTotalTiles, 1);
    uint32_t H = state.k;
    GlobalTensor<bfloat16_t> srcGlobalTensor;
    GlobalTensor<uint8_t> dstGlobalTensor;
    DataCopyParams xCopyInParams = {1U, static_cast<uint16_t>(H * sizeof(bfloat16_t)), 0U, 0U};
    DataCopyPadParams xCopyInPadParams{true, 0, 0, 0};
    DataCopyExtParams xCopyOutParams = {1U,
        static_cast<uint32_t>(state.mxQuantTokenAlignBytes + state.mxQuantScaleAlignBytes), 0U, 0U, 0U};
    SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
    for (int32_t index = 0; index < currentNum; index++) {
        srcGlobalTensor.SetGlobalBuffer((__gm__ bfloat16_t*)(state.params.aGmAddr +
            (currentOffset + index) * H * sizeof(bfloat16_t)));
        dstGlobalTensor.SetGlobalBuffer((__gm__ uint8_t*)(state.params.peermemInfo.quantTokenScalePtr +
            (currentOffset + index) * state.mxQuantTokenScaleAlignBytes));
        auto event = (index % DOUBLE_BUFFER == 0) ? EVENT_ID0 : EVENT_ID1;
        auto xInTensor = (index % DOUBLE_BUFFER == 0) ? state.xInTensor1_ : state.xInTensor2_;
        auto xOutTensor = (index % DOUBLE_BUFFER == 0) ? state.xOutTensor1_ : state.xOutTensor2_;
        WaitFlag<AscendC::HardEvent::MTE3_MTE2>(event);
        DataCopyPad(xInTensor, srcGlobalTensor, xCopyInParams, xCopyInPadParams);
        SetFlag<AscendC::HardEvent::MTE2_V>(event);
        WaitFlag<AscendC::HardEvent::MTE2_V>(event);
        __ubuf__ bfloat16_t* srcAddr = (__ubuf__ bfloat16_t*)xInTensor.GetPhyAddr();
        __ubuf__ uint16_t* maxExpAddr = (__ubuf__ uint16_t*)state.mxTempTensor_.GetPhyAddr();
        __ubuf__ uint16_t* halfScaleAddr = (__ubuf__ uint16_t*)state.mxTempTensor_[Ops::Base::CeilAlign(
            state.mxQuantScaleNumAlignPerToken, static_cast<uint32_t>(ALIGN_32))].GetPhyAddr();
        __ubuf__ int8_t* outDataAddr = (__ubuf__ int8_t*)xOutTensor.GetPhyAddr();
        __ubuf__ uint16_t* mxScaleAddr = (__ubuf__ uint16_t*)xOutTensor[state.mxQuantTokenAlignBytes_].GetPhyAddr();

        quant::ComputeMaxExp(srcAddr, maxExpAddr, H);
        quant::ComputeScale<QuantOutType>(maxExpAddr, mxScaleAddr, halfScaleAddr,
            state.mxQuantScaleNumAlignPerToken);
        if constexpr (QuantMode == E2M1_QUANT) {
            quant::ComputeFp4Data<bfloat16_t, QuantOutType, AscendC::RoundMode::CAST_TRUNC,
                AscendC::RoundMode::CAST_RINT>(srcAddr, halfScaleAddr, outDataAddr, H);
        } else {
            quant::ComputeFp8Data<bfloat16_t, QuantOutType, AscendC::RoundMode::CAST_TRUNC,
                AscendC::RoundMode::CAST_RINT>(srcAddr, halfScaleAddr, outDataAddr, H);
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

template <typename ActivationType, typename QuantOutType, typename QuantScaleOutType, int32_t QuantMode>
__aicore__ inline void CopyGMToGMPerToken(
    TokenDispatchState<ActivationType, QuantOutType, QuantMode> &state, const TileInfo &tile,
    int32_t rowDstOffsetInCore, int32_t remoteRankIdx, int32_t copyStartIdx, int32_t copyNum)
{
    if (copyNum <= 0) {
        return;
    }
    constexpr int32_t BufferNum = DISPATCH_BUFFER_NUM;
    constexpr TEventID kBufEvents[BufferNum] = {EVENT_ID0, EVENT_ID1, EVENT_ID2, EVENT_ID3, EVENT_ID4, EVENT_ID5};
    int64_t widthA = state.k / state.A_ELEMS_PER_BYTE;
    int64_t widthAScale = Ops::Base::CeilDiv(static_cast<int64_t>(state.k), static_cast<int64_t>(MXFP_DIVISOR_SIZE)) *
        MXFP_MULTI_BASE_SIZE;
    uint32_t copyInNum = Ops::Base::CeilAlign(static_cast<int64_t>(state.mxQuantTokenAlignBytes + state.mxQuantScaleAlignBytes),
        static_cast<int64_t>(ALIGN_32));
    GlobalTensor<ActivationType> remoteRankGlobalTensor;
    GlobalTensor<ActivationType> tokenRevGlobalTensor;
    GlobalTensor<QuantScaleOutType> scaleRevGlobalTensor;
    tokenRevGlobalTensor.SetGlobalBuffer(reinterpret_cast<__gm__ ActivationType*>(
        state.params.workspaceInfo.dispatchRevDataPtr + rowDstOffsetInCore * widthA));
    scaleRevGlobalTensor.SetGlobalBuffer(reinterpret_cast<__gm__ QuantScaleOutType *>(
        state.params.workspaceInfo.dispatchRevScalePtr + rowDstOffsetInCore * widthAScale));
    remoteRankGlobalTensor.SetGlobalBuffer(reinterpret_cast<__gm__ ActivationType*>(
        GetRankWinAddrWithOffset(remoteRankIdx, state.quantWinOffset)));

    for (int32_t bufferIdx = 0; bufferIdx < BufferNum; ++bufferIdx) {
        SetFlag<AscendC::HardEvent::MTE3_MTE2>(kBufEvents[bufferIdx]);
    }
    PipeBarrier<PIPE_ALL>();

    for (int32_t i = 0; i < copyNum; ++i) {
        int32_t topkIndex = state.validTopkIndexTensor_.GetValue(copyStartIdx + i);
        int32_t tokenIndex = topkIndex / state.topK;
        state.tripleTensor_[i * INT32_PER_256B].SetValue(RANK_ID, remoteRankIdx);
        state.tripleTensor_[i * INT32_PER_256B].SetValue(TOKEN_ID, tokenIndex);
        state.tripleTensor_[i * INT32_PER_256B].SetValue(TOPK_INDEX, topkIndex % state.topK);
    }

    int32_t primeCount = (copyNum < BufferNum) ? copyNum : BufferNum;
    for (int32_t primeIdx = 0; primeIdx < primeCount; ++primeIdx) {
        int32_t tokenIndex = state.tripleTensor_[primeIdx * INT32_PER_256B].GetValue(TOKEN_ID);
        uint64_t remoteCopyOffset = static_cast<uint64_t>(tokenIndex) * static_cast<uint64_t>(copyInNum);
        TEventID eventId = kBufEvents[primeIdx];
        WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventId);
        DataCopy(state.copyTmpTensors_[primeIdx], remoteRankGlobalTensor[remoteCopyOffset], copyInNum);
        SetFlag<AscendC::HardEvent::MTE2_MTE3>(eventId);
    }
    for (int32_t copyIdx = 0; copyIdx < copyNum; ++copyIdx) {
        int32_t outIdx = copyIdx % BufferNum;
        TEventID eventId = kBufEvents[outIdx];
        WaitFlag<AscendC::HardEvent::MTE2_MTE3>(eventId);

        LocalTensor<ActivationType> tokenScalebuf = state.copyTmpTensors_[outIdx];
        LocalTensor<QuantScaleOutType> bufScale =
            tokenScalebuf[state.mxQuantTokenAlignBytes].template ReinterpretCast<QuantScaleOutType>();
        DataCopyPad(tokenRevGlobalTensor[copyIdx * widthA], tokenScalebuf,
            {1, static_cast<uint16_t>(widthA * sizeof(ActivationType)), 0U, 0U, 0U});
        DataCopyPad(scaleRevGlobalTensor[copyIdx * widthAScale], bufScale,
            {1, static_cast<uint16_t>(widthAScale * sizeof(QuantScaleOutType)), 0U, 0U, 0U});
        SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventId);

        int32_t nextIdx = copyIdx + BufferNum;
        if (nextIdx < copyNum) {
            int32_t tokenIndex = state.tripleTensor_[nextIdx * INT32_PER_256B].GetValue(TOKEN_ID);
            uint64_t remoteCopyOffset = static_cast<uint64_t>(tokenIndex) * static_cast<uint64_t>(copyInNum);
            WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventId);
            DataCopy(state.copyTmpTensors_[outIdx], remoteRankGlobalTensor[remoteCopyOffset], copyInNum);
            SetFlag<AscendC::HardEvent::MTE2_MTE3>(eventId);
        }
    }
    for (int32_t bufferIdx = 0; bufferIdx < BufferNum; ++bufferIdx) {
        WaitFlag<AscendC::HardEvent::MTE3_MTE2>(kBufEvents[bufferIdx]);
    }
    SyncFuncStatic<AscendC::HardEvent::S_MTE3, SYNC_EVENT_ID3>();
    GlobalTensor<int32_t> tripleGlobalTensor;
    tripleGlobalTensor.SetGlobalBuffer((__gm__ int32_t*)state.params.workspaceInfo.tripleInfoPtr);
    DataCopy(tripleGlobalTensor[rowDstOffsetInCore * INT32_PER_256B], state.tripleTensor_, copyNum * INT32_PER_256B);
}

template <typename ActivationType, typename QuantOutType, typename QuantScaleOutType, int32_t QuantMode,
          bool ENABLE_A8W4>
__aicore__ inline void TripleInfoCalAndDispatch(
    TokenDispatchState<ActivationType, QuantOutType, QuantMode> &state, const TileInfo &tile,
    GMMAddrInfo &gmmAddrInfo, int32_t localExpertId)
{
    constexpr int32_t L1_TILE_M_I32 = static_cast<int32_t>(MegaMoeImpl::L1_TILE_M_256);
    int32_t priorExpertCumsum = (localExpertId == 0) ? 0 :
        state.cumsumInfoTensor_.GetValue(localExpertId * state.worldSize - 1);
    uint32_t topkIndexTensorSize =
        Ops::Base::CeilAlign(static_cast<int64_t>(state.sendTotalNum * sizeof(int32_t)), static_cast<int64_t>(ALIGN_32));
    if constexpr (ENABLE_A8W4) {
        if (localExpertId != 0) {
            CreateVecIndex(state.topkIndexTensor_, 0, topkIndexTensorSize / sizeof(int32_t));
            AscendC::PipeBarrier<PIPE_V>();
        }
    }
    for (uint32_t idx = tile.tileIdx; idx < state.worldSize * state.blockNumPerRank; idx += tile.totalTiles) {
        uint32_t dstRankIdx = idx / state.blockNumPerRank;
        uint64_t sendToCurExpTokenCnt = 0;
        GatherMask(state.validTopkIndexTensor_, state.topkIndexTensor_,
            state.gatherMaskInt32Tensor_[dstRankIdx * (state.maskSlotSize / sizeof(uint32_t))],
            true, state.sendTotalNum, {1, 1, 0, 0}, sendToCurExpTokenCnt);
        uint32_t aivIdxInGroup = idx % state.blockNumPerRank;
        int32_t rowStartIdxInDst = ((dstRankIdx == 0 && localExpertId == 0) ? 0 :
            state.cumsumInfoTensor_.GetValue(localExpertId * state.worldSize + dstRankIdx - 1));
        int32_t rowsThisCore = 0;
        int32_t rowDstOffsetInCore = 0;
        if (rowStartIdxInDst < state.maxOutputSize) {
            int32_t rowsCopyCnt = sendToCurExpTokenCnt;
            if (rowStartIdxInDst + rowsCopyCnt > state.maxOutputSize) {
                rowsCopyCnt = state.maxOutputSize - rowStartIdxInDst;
            }
            int32_t rowsNumInCore = Ops::Base::CeilDiv(rowsCopyCnt, static_cast<int32_t>(state.blockNumPerRank));
            int32_t rowSrcIdxInCore = aivIdxInGroup * rowsNumInCore;
            rowDstOffsetInCore = rowStartIdxInDst + rowSrcIdxInCore;
            if (rowDstOffsetInCore + rowsNumInCore > rowStartIdxInDst + rowsCopyCnt) {
                rowsNumInCore = rowStartIdxInDst + rowsCopyCnt - rowDstOffsetInCore;
            }
            if (rowsNumInCore > 0) {
                uint32_t tripleTensorAddr = state.ubBufferUsedAddr;
                uint32_t tripleTensorSize = rowsNumInCore * ALIGN_32;
                state.tripleTensor_ = LocalTensor<int32_t>(TPosition::VECCALC, tripleTensorAddr,
                    tripleTensorSize / sizeof(int32_t));
                SyncFuncStatic<AscendC::HardEvent::V_S, SYNC_EVENT_ID4>();
                CopyGMToGMPerToken<ActivationType, QuantOutType, QuantScaleOutType, QuantMode>(
                    state, tile, rowDstOffsetInCore, dstRankIdx, rowSrcIdxInCore, rowsNumInCore);
                rowsThisCore = rowsNumInCore;
            }
        }

        if (rowsThisCore > 0) {
            SyncFuncStatic<AscendC::HardEvent::MTE3_S, SYNC_EVENT_ID5>();
            int32_t rowStartLocal = rowDstOffsetInCore - priorExpertCumsum;
            int32_t rowEndLocal = rowStartLocal + rowsThisCore;
            int32_t waveLo = rowStartLocal / L1_TILE_M_I32;
            int32_t waveHi = (rowEndLocal - 1) / L1_TILE_M_I32;
            __gm__ int32_t* flagBase = gmmAddrInfo.dispatchToGmm1Flag;
            for (int32_t w = waveLo; w <= waveHi; ++w) {
                int32_t waveStartLocal = w * L1_TILE_M_I32;
                int32_t waveEndLocal = waveStartLocal + L1_TILE_M_I32;
                int32_t lo = rowStartLocal > waveStartLocal ? rowStartLocal : waveStartLocal;
                int32_t hi = rowEndLocal < waveEndLocal ? rowEndLocal : waveEndLocal;
                AtomicAdd(flagBase + w, int32_t(hi - lo));
            }
        }
    }
}

template <typename ActivationType, typename QuantOutType, int32_t QuantMode, bool ENABLE_A8W4>
__aicore__ inline void ExpertTokenNumCopyOut(
    TokenDispatchState<ActivationType, QuantOutType, QuantMode> &state, const TileInfo &tile,
    GlobalTensor<int32_t> &expertTokenNumsOut, GlobalTensor<int32_t> &cumsumInfoGlobalTensor)
{
    if constexpr (ENABLE_A8W4) {
        DataCopyPad(state.cumsumInfoTensor_, cumsumInfoGlobalTensor,
                {1U, static_cast<uint32_t>(state.worldSize * state.expertPerRank * sizeof(int32_t)), 0U, 0U, 0U},
                {true, 0U, 0U, 0U});
        AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(0);
    }
    int32_t lastRankIdx = static_cast<int32_t>(state.worldSize - 1);
    state.expertTokenNumsOutTensor_.SetValue(0, state.cumsumInfoTensor_.GetValue(lastRankIdx));
    for (int32_t expertIdx = 1; expertIdx < state.expertPerRank; expertIdx++) {
        int32_t cur = state.cumsumInfoTensor_.GetValue(expertIdx * static_cast<int32_t>(state.worldSize) + lastRankIdx);
        int32_t prev = state.cumsumInfoTensor_.GetValue((expertIdx - 1) * static_cast<int32_t>(state.worldSize) + lastRankIdx);
        state.expertTokenNumsOutTensor_.SetValue(expertIdx, cur - prev);
    }
    SyncFuncStatic<AscendC::HardEvent::S_MTE3, SYNC_EVENT_ID2>();
    DataCopyExtParams copyParams{1U, static_cast<uint32_t>(state.expertPerRank * sizeof(int32_t)), 0U, 0U, 0U};
    DataCopyPad(expertTokenNumsOut, state.expertTokenNumsOutTensor_, copyParams);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(0);
}

} // namespace TokenDispatch
} // namespace MegaMoeModules

#endif
