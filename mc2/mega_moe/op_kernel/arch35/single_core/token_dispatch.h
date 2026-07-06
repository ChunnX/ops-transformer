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
 * \brief Single-core token dispatch module: ExpertTokenNumCopyOut, QuantProcessInRank,
 *        CopyGMToGMPerToken, TripleInfoCalAndDispatch. All multi-core primitives
 *        (GetBlockIdx, GetBlockNum, GetSubBlockIdx, blockIdx_, aivCoreIdx_,
 *        subBlockIdx_, blockAivNum_, blockNum_) are replaced by explicit TileInfo
 *        parameter and explicit function parameters.
 */

#ifndef MEGA_MOE_SINGLE_CORE_TOKEN_DISPATCH_H
#define MEGA_MOE_SINGLE_CORE_TOKEN_DISPATCH_H

#include "mega_moe_tile_info.h"
#include "mega_moe_base.h"
#include "mega_moe_workspace_info.h"
#if __has_include("../../../moe_distribute_dispatch_v2/quantize_functions.h")
#include "../../../moe_distribute_dispatch_v2/quantize_functions.h"
#else
#include "../../../../moe_distribute_dispatch_v2/op_kernel/quantize_functions.h"
#endif
#if __has_include("../../../common/mc2_kernel_utils.h")
#include "../../../common/mc2_kernel_utils.h"
#else
#include "../../../../common/op_kernel/mc2_kernel_utils.h"
#endif

using namespace AscendC;

namespace TokenDispatch {

constexpr int32_t DISPATCH_BUFFER_NUM = 6;

template <typename ActivationType, typename QuantOutType, typename QuantScaleOutType,
          int32_t QuantMode, bool ENABLE_A8W4>
__aicore__ inline void ExpertTokenNumCopyOut(
    uint32_t worldSize, uint32_t expertPerRank,
    LocalTensor<int32_t> &cumsumInfoTensor,
    GlobalTensor<int32_t> &cumsumInfoGlobalTensor,
    LocalTensor<int32_t> &expertTokenNumsOutTensor,
    GlobalTensor<int32_t> &expertTokenNumsOut)
{
    if constexpr (ENABLE_A8W4) {
        DataCopyPad(cumsumInfoTensor, cumsumInfoGlobalTensor,
                {1U, static_cast<uint32_t>(worldSize * expertPerRank * sizeof(int32_t)), 0U, 0U, 0U},
                {true, 0U, 0U, 0U});
        SetFlag<HardEvent::MTE2_S>(0);
        WaitFlag<HardEvent::MTE2_S>(0);
    }
    int32_t lastRankIdx = static_cast<int32_t>(worldSize - 1);
    expertTokenNumsOutTensor.SetValue(0, cumsumInfoTensor.GetValue(lastRankIdx));
    for (int32_t expertIdx = 1; expertIdx < expertPerRank; expertIdx++) {
        int32_t cur = cumsumInfoTensor.GetValue(expertIdx * static_cast<int32_t>(worldSize) + lastRankIdx);
        int32_t prev = cumsumInfoTensor.GetValue((expertIdx - 1) * static_cast<int32_t>(worldSize) + lastRankIdx);
        expertTokenNumsOutTensor.SetValue(expertIdx, cur - prev);
    }
    SyncFuncStatic<HardEvent::S_MTE3, SYNC_EVENT_ID2>();
    DataCopyExtParams copyParams{1U, static_cast<uint32_t>(expertPerRank * sizeof(int32_t)), 0U, 0U, 0U};
    DataCopyPad(expertTokenNumsOut, expertTokenNumsOutTensor, copyParams);
    SetFlag<HardEvent::MTE3_MTE2>(0);
    WaitFlag<HardEvent::MTE3_MTE2>(0);
}

template <typename ActivationType, typename QuantOutType, typename QuantScaleOutType,
          int32_t QuantMode, bool ENABLE_A8W4>
__aicore__ inline void QuantProcessInRank(
    MegaMoeModules::TileInfo tile,
    uint32_t m, uint32_t k,
    uint32_t mxQuantTokenAlignBytes, uint32_t mxQuantScaleAlignBytes,
    uint32_t mxQuantScaleNumAlignPerToken, uint32_t mxQuantTokenScaleAlignBytes,
    LocalTensor<bfloat16_t> &xInTensor1, LocalTensor<bfloat16_t> &xInTensor2,
    LocalTensor<ActivationType> &xOutTensor1, LocalTensor<ActivationType> &xOutTensor2,
    LocalTensor<uint16_t> &mxTempTensor,
    const Params &params)
{
    if constexpr(g_coreType == AIC) {
        return;
    }

    int32_t curentNum;
    int32_t curentOffset;
    MegaMoeModules::TilingByTile(m, curentNum, curentOffset, tile.aivTileIdx, tile.aivTotalTiles, 1);
    uint32_t H = k;
    GlobalTensor<bfloat16_t> srcGlobalTensor;
    GlobalTensor<uint8_t> dstGlobalTensor;
    DataCopyParams xCopyInParams = {1U, static_cast<uint16_t>(H * sizeof(bfloat16_t)), 0U, 0U};
    DataCopyPadParams xCopyInPadParams{true, 0, 0, 0};
    DataCopyExtParams xCopyOutParams = {1U,
        static_cast<uint32_t>(mxQuantTokenAlignBytes + mxQuantScaleAlignBytes), 0U, 0U, 0U};
    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
    for (int32_t index = 0; index < curentNum; index++) {
        srcGlobalTensor.SetGlobalBuffer((__gm__ bfloat16_t*)(params.aGmAddr +
            (curentOffset + index) * H * sizeof(bfloat16_t)));
        dstGlobalTensor.SetGlobalBuffer((__gm__ uint8_t*)(params.peermemInfo.quantTokenScalePtr +
            (curentOffset + index) * mxQuantTokenScaleAlignBytes));
        auto event = (index % DOUBLE_BUFFER == 0) ? EVENT_ID0 : EVENT_ID1;
        auto xInTensor = (index % DOUBLE_BUFFER == 0) ? xInTensor1 : xInTensor2;
        auto xOutTensor = (index % DOUBLE_BUFFER == 0) ? xOutTensor1 : xOutTensor2;
        WaitFlag<HardEvent::MTE3_MTE2>(event);
        DataCopyPad(xInTensor, srcGlobalTensor, xCopyInParams, xCopyInPadParams);
        SetFlag<HardEvent::MTE2_V>(event);
        WaitFlag<HardEvent::MTE2_V>(event);
        __ubuf__ bfloat16_t* srcAddr = (__ubuf__ bfloat16_t*)xInTensor.GetPhyAddr();
        __ubuf__ uint16_t* maxExpAddr = (__ubuf__ uint16_t*)mxTempTensor.GetPhyAddr();
        __ubuf__ uint16_t* halfScaleAddr = (__ubuf__ uint16_t*)mxTempTensor[Ops::Base::CeilAlign(
            mxQuantScaleNumAlignPerToken, static_cast<uint32_t>(ALIGN_32))].GetPhyAddr();
        __ubuf__ int8_t* outDataAddr = (__ubuf__ int8_t*)xOutTensor.GetPhyAddr();
        __ubuf__ uint16_t* mxScaleAddr = (__ubuf__ uint16_t*)xOutTensor[mxQuantTokenAlignBytes].GetPhyAddr();

        quant::ComputeMaxExp(srcAddr, maxExpAddr, H);
        quant::ComputeScale<QuantOutType>(maxExpAddr, mxScaleAddr, halfScaleAddr, mxQuantScaleNumAlignPerToken);
        if constexpr (QuantMode == E2M1_QUANT) {
            quant::ComputeFp4Data<bfloat16_t, QuantOutType, RoundMode::CAST_TRUNC,
                RoundMode::CAST_RINT>(srcAddr, halfScaleAddr, outDataAddr, H);
        } else {
            quant::ComputeFp8Data<bfloat16_t, QuantOutType, RoundMode::CAST_TRUNC,
                RoundMode::CAST_RINT>(srcAddr, halfScaleAddr, outDataAddr, H);
        }
        SetFlag<HardEvent::V_MTE3>(event);
        WaitFlag<HardEvent::V_MTE3>(event);
        auto xOutBytesTensor = xOutTensor.template ReinterpretCast<uint8_t>();
        DataCopyPad(dstGlobalTensor, xOutBytesTensor, xCopyOutParams);
        SetFlag<HardEvent::MTE3_MTE2>(event);
    }
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
}

template <typename ActivationType, typename QuantOutType, typename QuantScaleOutType,
          int32_t QuantMode, bool ENABLE_A8W4>
__aicore__ inline void CopyGMToGMPerToken(
    int32_t rowDstOffsetInCore, int32_t remoteRankIdx,
    int32_t copyStartIdx, int32_t copyNum,
    uint32_t k, uint32_t topK,
    uint32_t mxQuantTokenAlignBytes, uint32_t mxQuantScaleAlignBytes,
    uint64_t quantWinOffset,
    LocalTensor<int32_t> &validTopkIndexTensor,
    LocalTensor<int32_t> &tripleTensor,
    LocalTensor<ActivationType> (&copyTmpTensors)[DISPATCH_BUFFER_NUM],
    GlobalTensor<int32_t> &tripleGlobalTensor,
    const Params &params)
{
    if (copyNum <= 0) {
        return;
    }
    constexpr int32_t BufferNum = DISPATCH_BUFFER_NUM;
    constexpr TEventID kBufEvents[BufferNum] = {EVENT_ID0, EVENT_ID1, EVENT_ID2, EVENT_ID3, EVENT_ID4, EVENT_ID5};
    constexpr uint32_t aElemsPerByte = Std::IsSame<QuantOutType, fp4x2_e2m1_t>::value ? 2U : 1U;
    int64_t widthA = k / aElemsPerByte;
    int64_t widthAScale = Ops::Base::CeilDiv(static_cast<int64_t>(k), static_cast<int64_t>(MXFP_DIVISOR_SIZE)) *
        MXFP_MULTI_BASE_SIZE;
    uint32_t copyInNum = Ops::Base::CeilAlign(static_cast<int64_t>(mxQuantTokenAlignBytes + mxQuantScaleAlignBytes),
        static_cast<int64_t>(ALIGN_32));
    GlobalTensor<ActivationType> remoteRankGlobalTensor;
    GlobalTensor<ActivationType> tokenRevGlobalTensor;
    GlobalTensor<QuantScaleOutType> scaleRevGlobalTensor;
    tokenRevGlobalTensor.SetGlobalBuffer(reinterpret_cast<__gm__ ActivationType*>(
        params.workspaceInfo.dispatchRevDataPtr + rowDstOffsetInCore * widthA));
    scaleRevGlobalTensor.SetGlobalBuffer(reinterpret_cast<__gm__ QuantScaleOutType*>(
        params.workspaceInfo.dispatchRevScalePtr + rowDstOffsetInCore * widthAScale));
    remoteRankGlobalTensor.SetGlobalBuffer(reinterpret_cast<__gm__ ActivationType*>(
        GetRankWinAddrWithOffset(remoteRankIdx, quantWinOffset)));

    for (int32_t bufferIdx = 0; bufferIdx < BufferNum; ++bufferIdx) {
        SetFlag<HardEvent::MTE3_MTE2>(kBufEvents[bufferIdx]);
    }
    PipeBarrier<PIPE_ALL>();
    for (int32_t i = 0; i < copyNum; ++i) {
        int32_t topkIndex = validTopkIndexTensor.GetValue(copyStartIdx + i);
        int32_t tokenIndex = topkIndex / topK;
        tripleTensor[i * INT32_PER_256B].SetValue(RANK_ID, remoteRankIdx);
        tripleTensor[i * INT32_PER_256B].SetValue(TOKEN_ID, tokenIndex);
        tripleTensor[i * INT32_PER_256B].SetValue(TOPK_INDEX, topkIndex % topK);
    }

    int32_t primeCount = (copyNum < BufferNum) ? copyNum : BufferNum;
    for (int32_t primeIdx = 0; primeIdx < primeCount; ++primeIdx) {
        int32_t tokenIndex = tripleTensor[primeIdx * INT32_PER_256B].GetValue(TOKEN_ID);
        uint64_t remoteCopyOffset = static_cast<uint64_t>(tokenIndex) * static_cast<uint64_t>(copyInNum);
        TEventID eventId = kBufEvents[primeIdx];
        WaitFlag<HardEvent::MTE3_MTE2>(eventId);
        DataCopy(copyTmpTensors[primeIdx], remoteRankGlobalTensor[remoteCopyOffset], copyInNum);
        SetFlag<HardEvent::MTE2_MTE3>(eventId);
    }
    for (int32_t copyIdx = 0; copyIdx < copyNum; ++copyIdx) {
        int32_t outIdx = copyIdx % BufferNum;
        TEventID eventId = kBufEvents[outIdx];
        WaitFlag<HardEvent::MTE2_MTE3>(eventId);

        LocalTensor<ActivationType> tokenScalebuf = copyTmpTensors[outIdx];
        LocalTensor<QuantScaleOutType> bufScale =
            tokenScalebuf[mxQuantTokenAlignBytes].template ReinterpretCast<QuantScaleOutType>();
        DataCopyPad(tokenRevGlobalTensor[copyIdx * widthA], tokenScalebuf,
            {1, static_cast<uint16_t>(widthA * sizeof(ActivationType)), 0U, 0U, 0U});
        DataCopyPad(scaleRevGlobalTensor[copyIdx * widthAScale], bufScale,
            {1, static_cast<uint16_t>(widthAScale * sizeof(QuantScaleOutType)), 0U, 0U, 0U});
        SetFlag<HardEvent::MTE3_MTE2>(eventId);

        int32_t nextIdx = copyIdx + BufferNum;
        if (nextIdx < copyNum) {
            int32_t tokenIndex = tripleTensor[nextIdx * INT32_PER_256B].GetValue(TOKEN_ID);
            uint64_t remoteCopyOffset = static_cast<uint64_t>(tokenIndex) * static_cast<uint64_t>(copyInNum);
            WaitFlag<HardEvent::MTE3_MTE2>(eventId);
            DataCopy(copyTmpTensors[outIdx], remoteRankGlobalTensor[remoteCopyOffset], copyInNum);
            SetFlag<HardEvent::MTE2_MTE3>(eventId);
        }
    }
    for (int32_t bufferIdx = 0; bufferIdx < BufferNum; ++bufferIdx) {
        WaitFlag<HardEvent::MTE3_MTE2>(kBufEvents[bufferIdx]);
    }
    SyncFuncStatic<HardEvent::S_MTE3, SYNC_EVENT_ID3>();
    DataCopy(tripleGlobalTensor[rowDstOffsetInCore * INT32_PER_256B], tripleTensor, copyNum * INT32_PER_256B);
}

template <typename ActivationType, typename QuantOutType, typename QuantScaleOutType,
          int32_t QuantMode, bool ENABLE_A8W4>
__aicore__ inline void TripleInfoCalAndDispatch(
    GMMAddrInfo &gmmAddrInfo, int32_t localExpertId,
    uint32_t blockNumPerRank,
    MegaMoeModules::TileInfo tile,
    uint32_t worldSize, uint32_t expertPerRank,
    uint64_t maxOutputSize, uint64_t sendTotalNum,
    uint32_t topK, uint32_t k,
    uint32_t maskSlotSize,
    uint32_t mxQuantTokenAlignBytes, uint32_t mxQuantScaleAlignBytes,
    uint64_t quantWinOffset,
    uint32_t ubBufferUsedAddr,
    LocalTensor<int32_t> &cumsumInfoTensor,
    LocalTensor<uint32_t> &gatherMaskInt32Tensor,
    LocalTensor<int32_t> &validTopkIndexTensor,
    LocalTensor<int32_t> &topkIndexTensor,
    LocalTensor<ActivationType> (&copyTmpTensors)[DISPATCH_BUFFER_NUM],
    GlobalTensor<int32_t> &tripleGlobalTensor,
    const Params &params)
{
    constexpr int32_t L1_TILE_M_I32 = static_cast<int32_t>(MegaMoeImpl::L1_TILE_M_256);
    int32_t priorExpertCumsum = (localExpertId == 0) ? 0 :
        cumsumInfoTensor.GetValue(localExpertId * static_cast<int32_t>(worldSize) - 1);
    uint32_t topkIndexTensorSize =
        Ops::Base::CeilAlign(static_cast<int64_t>(sendTotalNum * sizeof(int32_t)), static_cast<int64_t>(ALIGN_32));
    if constexpr (ENABLE_A8W4) {
        if (localExpertId != 0) {
            CreateVecIndex(topkIndexTensor, 0, topkIndexTensorSize / sizeof(int32_t));
            PipeBarrier<PIPE_V>();
        }
    }
    for (uint32_t idx = tile.tileIdx; idx < worldSize * blockNumPerRank; idx += tile.totalTiles) {
        uint32_t dstRankIdx = idx / blockNumPerRank;
        uint64_t sendToCurExpTokenCnt = 0;
        GatherMask(validTopkIndexTensor, topkIndexTensor,
            gatherMaskInt32Tensor[dstRankIdx * (maskSlotSize / sizeof(uint32_t))],
            true, static_cast<uint32_t>(sendTotalNum), {1, 1, 0, 0}, sendToCurExpTokenCnt);
        uint32_t aivIdxInGroup = idx % blockNumPerRank;
        int32_t rowStartIdxInDst = ((dstRankIdx == 0 && localExpertId == 0) ? 0 :
            cumsumInfoTensor.GetValue(localExpertId * static_cast<int32_t>(worldSize) + static_cast<int32_t>(dstRankIdx) - 1));
        int32_t rowsThisCore = 0;
        int32_t rowDstOffsetInCore = 0;
        if (rowStartIdxInDst < maxOutputSize) {
            int32_t rowsCopyCnt = static_cast<int32_t>(sendToCurExpTokenCnt);
            if (rowStartIdxInDst + rowsCopyCnt > maxOutputSize) {
                rowsCopyCnt = maxOutputSize - rowStartIdxInDst;
            }
            int32_t rowsNumInCore = Ops::Base::CeilDiv(rowsCopyCnt, static_cast<int32_t>(blockNumPerRank));
            int32_t rowSrcIdxInCore = static_cast<int32_t>(aivIdxInGroup) * rowsNumInCore;
            rowDstOffsetInCore = rowStartIdxInDst + rowSrcIdxInCore;
            if (rowDstOffsetInCore + rowsNumInCore > rowStartIdxInDst + rowsCopyCnt) {
                rowsNumInCore = rowStartIdxInDst + rowsCopyCnt - rowDstOffsetInCore;
            }
            if (rowsNumInCore > 0) {
                uint32_t tripleTensorAddr = ubBufferUsedAddr;
                uint32_t tripleTensorSize = rowsNumInCore * ALIGN_32;
                LocalTensor<int32_t> localTripleTensor = LocalTensor<int32_t>(TPosition::VECCALC, tripleTensorAddr,
                    tripleTensorSize / sizeof(int32_t));
                SyncFuncStatic<HardEvent::V_S, SYNC_EVENT_ID4>();
                CopyGMToGMPerToken<ActivationType, QuantOutType, QuantScaleOutType, QuantMode, ENABLE_A8W4>(
                    rowDstOffsetInCore, static_cast<int32_t>(dstRankIdx), rowSrcIdxInCore, rowsNumInCore,
                    k, topK, mxQuantTokenAlignBytes, mxQuantScaleAlignBytes, quantWinOffset,
                    validTopkIndexTensor, localTripleTensor, copyTmpTensors, tripleGlobalTensor, params);
                rowsThisCore = rowsNumInCore;
            }
        }

        if (rowsThisCore > 0) {
            SyncFuncStatic<HardEvent::MTE3_S, SYNC_EVENT_ID5>();
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

} // namespace TokenDispatch

#endif // MEGA_MOE_SINGLE_CORE_TOKEN_DISPATCH_H
