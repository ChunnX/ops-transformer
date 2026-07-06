/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file combine_reduce.h
 * \brief Single-core CombineReduce module: BuffInit, Unpermute, CrossRankSyncInWorldSize.
 *        All core-identification replaced by TileInfo; no GetBlockIdx/GetBlockNum/GetSubBlockIdx.
 */

#ifndef MEGA_MOE_COMBINE_REDUCE_H
#define MEGA_MOE_COMBINE_REDUCE_H

#include "mega_moe_tile_info.h"
#include "mega_moe_base.h"
#include "mega_moe_workspace_info.h"
#include "mega_moe_combine_send.h"

using namespace AscendC;

namespace CombineReduce {

template <int32_t CombineQuantMode>
struct CombineReduceBuffers {
    LocalTensor<bfloat16_t> dataResTensor;
    LocalTensor<float> dataResFp32Tensor;
    LocalTensor<float> topKWeightsTensor;
    LocalTensor<bfloat16_t> bf16ScaleTensor;
    LocalTensor<float> fp32ScaleTensor;
};

template <int32_t CombineQuantMode, typename TopkWeightsType>
__aicore__ inline void BuffInit(CombineReduceBuffers<CombineQuantMode>& buf, const Params& params)
{
    uint32_t k = params.tilingData->h;
    uint32_t m = params.tilingData->bs;
    uint32_t topK = params.tilingData->topK;

    uint32_t dataResBufAlign = Ops::Base::CeilAlign(
        static_cast<uint32_t>(UNPERMUTE_LIST_NUM * k * sizeof(bfloat16_t)), static_cast<uint32_t>(ALIGN_32));
    uint32_t dataResFp32BufAlign = dataResBufAlign * HALF_TO_FP32;
    uint32_t topKWeightsBufAlign = Ops::Base::CeilAlign(
        static_cast<uint32_t>(m * topK * sizeof(float)), static_cast<uint32_t>(ALIGN_32));
    uint32_t tempBufAlign = Ops::Base::CeilAlign(
        static_cast<uint32_t>(m * topK * sizeof(bfloat16_t)), uint32_t(ALIGN_32));

    uint32_t dataResAddr = 0;
    uint32_t dataResSize = dataResBufAlign / sizeof(bfloat16_t);
    buf.dataResTensor = LocalTensor<bfloat16_t>(TPosition::VECCALC, dataResAddr, dataResSize);

    uint32_t dataResFp32Addr = dataResAddr + dataResBufAlign;
    uint32_t dataResFp32Size = dataResFp32BufAlign / sizeof(float);
    buf.dataResFp32Tensor = LocalTensor<float>(TPosition::VECCALC, dataResFp32Addr, dataResFp32Size);

    uint32_t topKWeightsAddr = dataResFp32Addr + dataResFp32BufAlign;
    uint32_t topKWeightsSize = topKWeightsBufAlign / sizeof(float);
    buf.topKWeightsTensor = LocalTensor<float>(TPosition::VECCALC, topKWeightsAddr, topKWeightsSize);

    uint32_t tempAddr = topKWeightsAddr + topKWeightsBufAlign;
    if constexpr (CombineQuantMode != COMBINE_NO_QUANT) {
        uint32_t scaleNum = Ops::Base::CeilAlign(static_cast<uint32_t>(k), static_cast<uint32_t>(ALIGN_32));
        uint32_t bf16ScaleBufAlign = Ops::Base::CeilAlign(static_cast<uint32_t>
            (scaleNum * sizeof(bfloat16_t) * DOUBLE_BUFFER * HALF_TO_FP32), static_cast<uint32_t>(ALIGN_32));
        buf.bf16ScaleTensor = LocalTensor<bfloat16_t>(
            TPosition::VECCALC, tempAddr, bf16ScaleBufAlign / sizeof(bfloat16_t));
        tempAddr += bf16ScaleBufAlign;
        uint32_t fp32ScaleBufAlign = Ops::Base::CeilAlign(static_cast<uint32_t>
            (scaleNum * sizeof(float) * DOUBLE_BUFFER * HALF_TO_FP32), static_cast<uint32_t>(ALIGN_32));
        buf.fp32ScaleTensor = LocalTensor<float>(TPosition::VECCALC, tempAddr, fp32ScaleBufAlign / sizeof(float));
        tempAddr += fp32ScaleBufAlign;
    }
    if constexpr (Std::IsSame<TopkWeightsType, float>::value) {
        GlobalTensor<float> topKWeightsGlobalTensor;
        topKWeightsGlobalTensor.SetGlobalBuffer((__gm__ float*)params.probsGmAddr);
        DataCopyExtParams copyParams = {1U, static_cast<uint32_t>(m * topK * sizeof(float)), 0U, 0U, 0U};
        DataCopyPadExtParams<float> copyPadParams{false, 0U, 0U, 0U};
        DataCopyPad(buf.topKWeightsTensor, topKWeightsGlobalTensor, copyParams, copyPadParams);
    }
    if constexpr (Std::IsSame<TopkWeightsType, bfloat16_t>::value) {
        uint32_t tempSize = tempBufAlign / sizeof(bfloat16_t);
        LocalTensor<bfloat16_t> tempLocal(TPosition::VECCALC, tempAddr, tempSize);
        GlobalTensor<bfloat16_t> topkWeightsGlobalTensor;
        topkWeightsGlobalTensor.SetGlobalBuffer((__gm__ bfloat16_t*)params.probsGmAddr);
        DataCopyExtParams copyParams = {1U, static_cast<uint32_t>(m * topK * sizeof(bfloat16_t)), 0U, 0U, 0U};
        DataCopyPadExtParams<bfloat16_t> copyPadParams{false, 0U, 0U, 0U};
        DataCopyPad(tempLocal, topkWeightsGlobalTensor, copyParams, copyPadParams);
        SyncFuncStatic<AscendC::HardEvent::MTE2_V, SYNC_EVENT_ID2>();
        Cast(buf.topKWeightsTensor, tempLocal, AscendC::RoundMode::CAST_NONE, m * topK);
    }
}

template <int32_t CombineQuantMode>
__aicore__ inline void Unpermute(const TileInfo& tile, CombineReduceBuffers<CombineQuantMode>& buf,
    const Params& params)
{
    uint32_t k = params.tilingData->h;
    uint32_t m = params.tilingData->bs;
    uint32_t topK = params.tilingData->topK;

    int32_t coreLen, coreOffset;
    TilingByTile(m, coreLen, coreOffset, tile.aivTileIdx, tile.aivTotalTiles, 1);

    GlobalTensor<bfloat16_t> expandedX;
    expandedX.SetGlobalBuffer((__gm__ bfloat16_t*)params.peermemInfo.combineSendPtr);
    GlobalTensor<bfloat16_t> output;
    output.SetGlobalBuffer((__gm__ bfloat16_t*)params.y2GmAddr);
    SetFlag<AscendC::HardEvent::V_MTE2>(0);
    SetFlag<AscendC::HardEvent::V_MTE2>(1);
    for (int32_t tokenIdx = coreOffset; tokenIdx < coreLen + coreOffset; tokenIdx++) {
        SyncFuncStatic<AscendC::HardEvent::MTE3_MTE2, SYNC_EVENT_ID2>();
        LocalTensor<bfloat16_t> dataIn0Bf16 = buf.dataResTensor[k];
        LocalTensor<bfloat16_t> dataIn1Bf16 = buf.dataResTensor[k * 2];
        LocalTensor<float> dataIn0Fp32 = buf.dataResFp32Tensor[k];
        LocalTensor<float> dataIn1Fp32 = buf.dataResFp32Tensor[k * 2];
        for (int32_t expId = 0; expId < topK; ++expId) {
            float expScale = buf.topKWeightsTensor.GetValue(tokenIdx * topK + expId);
            auto event = (expId % DOUBLE_BUFFER == 0) ? 0 : 1;
            auto dataInBf16 = (expId % DOUBLE_BUFFER == 0) ? dataIn0Bf16 : dataIn1Bf16;
            auto dataInFp32 = (expId % DOUBLE_BUFFER == 0) ? dataIn0Fp32 : dataIn1Fp32;
            if constexpr (CombineQuantMode == COMBINE_NO_QUANT) {
                WaitFlag<AscendC::HardEvent::V_MTE2>(event);
                DataCopy(dataInBf16, expandedX[(tokenIdx * topK + expId) * k], k);
                SetFlag<AscendC::HardEvent::MTE2_V>(event);
                WaitFlag<AscendC::HardEvent::MTE2_V>(event);
                SetFlag<AscendC::HardEvent::S_V>(event);
                WaitFlag<AscendC::HardEvent::S_V>(event);
                Cast(dataInFp32, dataInBf16, AscendC::RoundMode::CAST_NONE, k);
            } else {
                uint32_t nScale = Ops::Base::CeilDiv(k, uint32_t(MXFP_SCALE_GROUP_NUM));
                uint32_t quantTokenSize = k + nScale;
                uint32_t quantEleNum = quantTokenSize / sizeof(bfloat16_t);
                WaitFlag<AscendC::HardEvent::V_MTE2>(event);
                DataCopy(dataInBf16, expandedX[(tokenIdx * topK + expId) * quantEleNum], quantEleNum);
                SetFlag<AscendC::HardEvent::MTE2_V>(event);
                WaitFlag<AscendC::HardEvent::MTE2_V>(event);
                using Fp8Type = typename std::conditional<CombineQuantMode == MXFP8_E4M3_COMM_QUANT,
                    fp8_e4m3fn_t, fp8_e5m2_t>::type;
                MegaMoeCombineImpl::DeQuantMxFp8<Fp8Type, bfloat16_t>(dataInBf16, dataInFp32,
                    buf.bf16ScaleTensor, buf.fp32ScaleTensor, nScale, k);
            }
            PipeBarrier<PIPE_V>();
            if (expId == 0) {
                Muls(buf.dataResFp32Tensor, dataInFp32, expScale, k);
            } else {
                Muls(dataInFp32, dataInFp32, expScale, k);
                PipeBarrier<PIPE_V>();
                Add(buf.dataResFp32Tensor, buf.dataResFp32Tensor, dataInFp32, k);
                PipeBarrier<PIPE_V>();
            }
            SetFlag<AscendC::HardEvent::V_MTE2>(event);
        }
        Cast(buf.dataResTensor, buf.dataResFp32Tensor, AscendC::RoundMode::CAST_RINT, k);
        SyncFuncStatic<AscendC::HardEvent::V_MTE3, SYNC_EVENT_ID3>();
        DataCopy(output[tokenIdx * k], buf.dataResTensor, k);
    }
    WaitFlag<AscendC::HardEvent::V_MTE2>(0);
    WaitFlag<AscendC::HardEvent::V_MTE2>(1);
}

template <int32_t CombineQuantMode>
__aicore__ inline void CrossRankSyncInWorldSize(const TileInfo& tile, const Params& params)
{
    if constexpr(g_coreType == AIC) {
        return;
    }
    uint32_t worldSize = params.tilingData->epWorldSize;
    uint32_t rankId = params.combineCommParams.rankId;

    __gm__ int32_t* syncRank = (__gm__ int32_t*)(params.peermemInfo.rankSyncInWorldPtr);
    __gm__ int32_t* syncCount = (__gm__ int32_t*)(params.peermemInfo.rankSyncInWorldPtr +
        48 * 1024 + tile.aivTileIdx * 64);
    int count = ReadGmByPassDCache(syncCount) + 1;
    for (int i = tile.aivTileIdx; i < worldSize; i += tile.aivTotalTiles) {
        __gm__ int32_t* syncRemoteAddr = (__gm__ int32_t*)(winRankAddr_[i]) + rankId * 16;
        WriteGmByPassDCache(syncRemoteAddr, count);
        auto syncCheck = syncRank + i * 16;
        GmSignalWaitBarrier(syncCheck, count);
    }
    WriteGmByPassDCache(syncCount, count);
    PipeBarrier<PIPE_ALL>();
    SyncAll<true>();
}

} // namespace CombineReduce

#endif
