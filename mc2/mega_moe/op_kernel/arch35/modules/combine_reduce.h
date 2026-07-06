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
 * \file combine_reduce.h
 * \brief Single-core CombineReduce module: Unpermute (reduce tokens back with weights),
 *        cross-rank sync, and DeQuant operations.
 */

#ifndef MEGA_MOE_COMBINE_REDUCE_H
#define MEGA_MOE_COMBINE_REDUCE_H

#include "kernel_operator.h"
#include "mega_moe_tile_info.h"
#include "mega_moe_base.h"
#include "mega_moe_workspace_info.h"
#include "mega_moe_combine_send.h"

using namespace AscendC;

namespace MegaMoeModules {
namespace CombineReduce {

template <int32_t CombineQuantMode>
struct CombineReduceState {
    LocalTensor<bfloat16_t> dataResTensor_;
    LocalTensor<float> dataResFp32Tensor_;
    LocalTensor<float> topKWeightsTensor_;
    LocalTensor<float> fp32ScaleTensor_;
    LocalTensor<bfloat16_t> bf16ScaleTensor_;

    uint32_t m;
    uint32_t k;
    uint32_t topK;
    uint32_t rankId;
    uint32_t worldSize;
    uint32_t expertPerRank;
    Params params;
};

template <int32_t CombineQuantMode>
__aicore__ inline void BuffInit(
    CombineReduceState<CombineQuantMode> &state, const TileInfo &tile,
    const Params &params, uint32_t m, uint32_t k, uint32_t topK,
    uint32_t worldSize, uint32_t expertPerRank, uint32_t rankId,
    GlobalTensor<float> &topKWeightsGlobal, GlobalTensor<bfloat16_t> &topkWeightsBf16Global,
    typename std::conditional<Std::IsSame<typename std::conditional<
        (true), float, float>::type, float>::value, float, bfloat16_t>::type topkWeightsType)
{
    state.params = params;
    state.m = m;
    state.k = k;
    state.topK = topK;
    state.worldSize = worldSize;
    state.expertPerRank = expertPerRank;
    state.rankId = rankId;

    uint32_t dataResBufAlign = Ops::Base::CeilAlign(
        static_cast<uint32_t>(UNPERMUTE_LIST_NUM * k * sizeof(bfloat16_t)), static_cast<uint32_t>(ALIGN_32));
    uint32_t dataResFp32BufAlign = dataResBufAlign * HALF_TO_FP32;
    uint32_t topKWeightsBufAlign = Ops::Base::CeilAlign(
        static_cast<uint32_t>(m * topK * sizeof(float)), static_cast<uint32_t>(ALIGN_32));

    uint32_t dataResAddr = 0;
    uint32_t dataResSize = dataResBufAlign / sizeof(bfloat16_t);
    state.dataResTensor_ = LocalTensor<bfloat16_t>(TPosition::VECCALC, dataResAddr, dataResSize);
    uint32_t dataResFp32Addr = dataResAddr + dataResBufAlign;
    uint32_t dataResFp32Size = dataResFp32BufAlign / sizeof(float);
    state.dataResFp32Tensor_ = LocalTensor<float>(TPosition::VECCALC, dataResFp32Addr, dataResFp32Size);
    uint32_t topKWeightsAddr = dataResFp32Addr + dataResFp32BufAlign;
    uint32_t topKWeightsSize = topKWeightsBufAlign / sizeof(float);
    state.topKWeightsTensor_ = LocalTensor<float>(TPosition::VECCALC, topKWeightsAddr, topKWeightsSize);
    uint32_t tempAddr = topKWeightsAddr + topKWeightsBufAlign;
    if constexpr (CombineQuantMode != COMBINE_NO_QUANT) {
        uint32_t scaleNum = Ops::Base::CeilAlign(static_cast<uint32_t>(k), static_cast<uint32_t>(ALIGN_32));
        uint32_t bf16ScaleBufAlign = Ops::Base::CeilAlign(static_cast<uint32_t>
            (scaleNum * sizeof(bfloat16_t) * DOUBLE_BUFFER * HALF_TO_FP32), static_cast<uint32_t>(ALIGN_32));
        state.bf16ScaleTensor_ = LocalTensor<bfloat16_t>(
            TPosition::VECCALC, tempAddr, bf16ScaleBufAlign / sizeof(bfloat16_t));
        tempAddr += bf16ScaleBufAlign;
        uint32_t fp32ScaleBufAlign = Ops::Base::CeilAlign(static_cast<uint32_t>
            (scaleNum * sizeof(float) * DOUBLE_BUFFER * HALF_TO_FP32), static_cast<uint32_t>(ALIGN_32));
        state.fp32ScaleTensor_ = LocalTensor<float>(TPosition::VECCALC, tempAddr, fp32ScaleBufAlign / sizeof(float));
        tempAddr += fp32ScaleBufAlign;
    }
}

template <int32_t CombineQuantMode>
__aicore__ inline void CrossRankSyncInWorldSize(
    CombineReduceState<CombineQuantMode> &state, const TileInfo &tile)
{
    if constexpr(g_coreType == AIC) {
        return;
    }
    __gm__ int32_t* syncRank = (__gm__ int32_t*)(state.params.peermemInfo.rankSyncInWorldPtr);
    __gm__ int32_t* syncCount = (__gm__ int32_t*)(state.params.peermemInfo.rankSyncInWorldPtr +
        48 * 1024 + tile.aivTileIdx * 64);
    int count = ReadGmByPassDCache(syncCount) + 1;
    for (int i = tile.aivTileIdx; i < state.worldSize; i += tile.aivTotalTiles) {
        __gm__ int32_t* syncRemoteAddr = (__gm__ int32_t*)(winRankAddr_[i]) + state.rankId * 16;
        WriteGmByPassDCache(syncRemoteAddr, count);
        auto syncCheck = syncRank + i * 16;
        GmSignalWaitBarrier(syncCheck, count);
    }
    WriteGmByPassDCache(syncCount, count);
    PipeBarrier<PIPE_ALL>();
    SyncAll<true>();
}

template <int32_t CombineQuantMode>
__aicore__ inline void Unpermute(
    CombineReduceState<CombineQuantMode> &state, const TileInfo &tile)
{
    int32_t coreLen, coreOffset;
    TilingByTile(state.m, coreLen, coreOffset, tile.aivTileIdx, tile.aivTotalTiles, 1);
    GlobalTensor<bfloat16_t> expandedX;
    expandedX.SetGlobalBuffer((__gm__ bfloat16_t*)state.params.peermemInfo.combineSendPtr);
    GlobalTensor<bfloat16_t> output;
    output.SetGlobalBuffer((__gm__ bfloat16_t*)state.params.y2GmAddr);
    SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
    SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
    for (int32_t tokenIdx = coreOffset; tokenIdx < coreLen + coreOffset; tokenIdx++) {
        SyncFuncStatic<AscendC::HardEvent::MTE3_MTE2, SYNC_EVENT_ID2>();
        LocalTensor<bfloat16_t> dataIn0Bf16 = state.dataResTensor_[state.k];
        LocalTensor<bfloat16_t> dataIn1Bf16 = state.dataResTensor_[state.k * 2];
        LocalTensor<float> dataIn0Fp32 = state.dataResFp32Tensor_[state.k];
        LocalTensor<float> dataIn1Fp32 = state.dataResFp32Tensor_[state.k * 2];
        for (int32_t expId = 0; expId < state.topK; ++expId) {
            float expScale = state.topKWeightsTensor_.GetValue(tokenIdx * state.topK + expId);
            auto event = (expId % DOUBLE_BUFFER == 0) ? EVENT_ID0 : EVENT_ID1;
            auto dataInBf16 = (expId % DOUBLE_BUFFER == 0) ? dataIn0Bf16 : dataIn1Bf16;
            auto dataInFp32 = (expId % DOUBLE_BUFFER == 0) ? dataIn0Fp32 : dataIn1Fp32;
            if constexpr (CombineQuantMode == COMBINE_NO_QUANT) {
                WaitFlag<AscendC::HardEvent::V_MTE2>(event);
                DataCopy(dataInBf16, expandedX[(tokenIdx * state.topK + expId) * state.k], state.k);
                SetFlag<AscendC::HardEvent::MTE2_V>(event);
                WaitFlag<AscendC::HardEvent::MTE2_V>(event);
                SetFlag<AscendC::HardEvent::S_V>(event);
                WaitFlag<AscendC::HardEvent::S_V>(event);
                Cast(dataInFp32, dataInBf16, AscendC::RoundMode::CAST_NONE, state.k);
            } else {
                uint32_t nScale = Ops::Base::CeilDiv(state.k, uint32_t(MXFP_SCALE_GROUP_NUM));
                uint32_t quantTokenSize = state.k + nScale;
                uint32_t quantEleNum = quantTokenSize / sizeof(bfloat16_t);
                WaitFlag<AscendC::HardEvent::V_MTE2>(event);
                DataCopy(dataInBf16, expandedX[(tokenIdx * state.topK + expId) * quantEleNum], quantEleNum);
                SetFlag<AscendC::HardEvent::MTE2_V>(event);
                WaitFlag<AscendC::HardEvent::MTE2_V>(event);
                using Fp8Type = typename std::conditional<CombineQuantMode == MXFP8_E4M3_COMM_QUANT,
                    fp8_e4m3fn_t, fp8_e5m2_t>::type;
                MegaMoeCombineImpl::DeQuantMxFp8<Fp8Type, bfloat16_t>(dataInBf16, dataInFp32,
                    state.bf16ScaleTensor_, state.fp32ScaleTensor_, nScale, state.k);
            }
            PipeBarrier<PIPE_V>();
            if (expId == 0) {
                Muls(state.dataResFp32Tensor_, dataInFp32, expScale, state.k);
            } else {
                Muls(dataInFp32, dataInFp32, expScale, state.k);
                PipeBarrier<PIPE_V>();
                Add(state.dataResFp32Tensor_, state.dataResFp32Tensor_, dataInFp32, state.k);
                PipeBarrier<PIPE_V>();
            }
            SetFlag<AscendC::HardEvent::V_MTE2>(event);
        }
        Cast(state.dataResTensor_, state.dataResFp32Tensor_, AscendC::RoundMode::CAST_RINT, state.k);
        SyncFuncStatic<AscendC::HardEvent::V_MTE3, SYNC_EVENT_ID3>();
        DataCopy(output[tokenIdx * state.k], state.dataResTensor_, state.k);
    }
    WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
    WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
}

} // namespace CombineReduce
} // namespace MegaMoeModules

#endif
