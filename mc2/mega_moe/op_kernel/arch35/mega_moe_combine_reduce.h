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
 * \file mega_moe_combine_reduce.h
 * \brief Single-core combine reduce implementation.
 */

#ifndef MEGA_MOE_COMBINE_REDUCE_H
#define MEGA_MOE_COMBINE_REDUCE_H

#include "kernel_operator.h"
#include "mega_moe_base.h"
#include "mega_moe_combine_send.h"

using namespace AscendC;

namespace MegaMoeCombineReduceImpl {
struct CombineReduceTileSchedule {
    uint32_t tileId;
    uint32_t tileNum;
};

struct CombineReduceShapeContext {
    uint32_t m;
    uint32_t k;
    uint32_t topK;
};

struct CombineReduceBufferContext {
    GM_ADDR expandedXAddr;
    GM_ADDR topKWeightsAddr;
    GM_ADDR outputAddr;
};

struct CombineReduceSyncContext {
    GM_ADDR rankSyncAddr;
    uint32_t rankId;
    uint32_t worldSize;
};

struct CombineReduceWorkspace {
    LocalTensor<bfloat16_t> dataResTensor;
    LocalTensor<float> dataResFp32Tensor;
    LocalTensor<float> topKWeightsTensor;
    LocalTensor<bfloat16_t> topKWeightsBf16Tensor;
    LocalTensor<bfloat16_t> bf16ScaleTensor;
    LocalTensor<float> fp32ScaleTensor;
};

__aicore__ inline void ComputeCombineReduceTileRange(const CombineReduceTileSchedule& tileSchedule,
    uint32_t totalLength, int32_t& tileLength, int32_t& tileOffset)
{
    tileLength = 0;
    tileOffset = 0;
    if (tileSchedule.tileNum == 0 || tileSchedule.tileId >= tileSchedule.tileNum) {
        return;
    }
    uint32_t lengthPerTile = Ops::Base::CeilDiv(totalLength, tileSchedule.tileNum);
    uint32_t offset = tileSchedule.tileId * lengthPerTile;
    if (offset >= totalLength) {
        return;
    }
    tileOffset = static_cast<int32_t>(offset);
    tileLength = static_cast<int32_t>((lengthPerTile < totalLength - offset) ?
        lengthPerTile : totalLength - offset);
}

template <int32_t CombineQuantMode, typename TopkWeightsType>
__aicore__ inline int32_t InitCombineReduceWorkspace(int32_t tileLength,
    const CombineReduceShapeContext& shapeContext, CombineReduceWorkspace& workspace)
{
    uint32_t dataResBufAlign = Ops::Base::CeilAlign(
        static_cast<uint32_t>(UNPERMUTE_LIST_NUM * shapeContext.k * sizeof(bfloat16_t)),
        static_cast<uint32_t>(ALIGN_32));
    uint32_t dataResFp32BufAlign = dataResBufAlign * HALF_TO_FP32;
    uint32_t dataResAddr = 0;
    workspace.dataResTensor = LocalTensor<bfloat16_t>(TPosition::VECCALC, dataResAddr,
        dataResBufAlign / sizeof(bfloat16_t));
    uint32_t dataResFp32Addr = dataResAddr + dataResBufAlign;
    workspace.dataResFp32Tensor = LocalTensor<float>(TPosition::VECCALC, dataResFp32Addr,
        dataResFp32BufAlign / sizeof(float));
    uint32_t tempAddr = dataResFp32Addr + dataResFp32BufAlign;

    int32_t batchLength = UNPERMUTE_BASE_TOKENS_PER_BATCH;
    if (shapeContext.topK > static_cast<uint32_t>(UNPERMUTE_BASE_TOPK)) {
        batchLength = batchLength * UNPERMUTE_BASE_TOPK / shapeContext.topK;
    }
    if (batchLength > tileLength) {
        batchLength = tileLength;
    }
    uint32_t topKWeightsBufAlign = Ops::Base::CeilAlign(
        static_cast<uint32_t>(batchLength * shapeContext.topK * sizeof(float)),
        static_cast<uint32_t>(ALIGN_32));
    workspace.topKWeightsTensor = LocalTensor<float>(TPosition::VECCALC, tempAddr,
        topKWeightsBufAlign / sizeof(float));
    tempAddr += topKWeightsBufAlign;

    if constexpr (Std::IsSame<TopkWeightsType, bfloat16_t>::value) {
        uint32_t tempBufAlign = Ops::Base::CeilAlign(
            static_cast<uint32_t>(batchLength * shapeContext.topK * sizeof(bfloat16_t)),
            static_cast<uint32_t>(ALIGN_32));
        workspace.topKWeightsBf16Tensor = LocalTensor<bfloat16_t>(TPosition::VECCALC, tempAddr,
            tempBufAlign / sizeof(bfloat16_t));
        tempAddr += tempBufAlign;
    }

    if constexpr (CombineQuantMode != COMBINE_NO_QUANT) {
        uint32_t scaleNum = Ops::Base::CeilDiv(shapeContext.k, static_cast<uint32_t>(ALIGN_32));
        uint32_t bf16ScaleBufAlign = Ops::Base::CeilAlign(static_cast<uint32_t>(
            scaleNum * sizeof(bfloat16_t) * DOUBLE_BUFFER * HALF_TO_FP32), static_cast<uint32_t>(ALIGN_32));
        workspace.bf16ScaleTensor = LocalTensor<bfloat16_t>(TPosition::VECCALC, tempAddr,
            bf16ScaleBufAlign / sizeof(bfloat16_t));
        tempAddr += bf16ScaleBufAlign;
        uint32_t fp32ScaleBufAlign = Ops::Base::CeilAlign(static_cast<uint32_t>(
            scaleNum * sizeof(float) * DOUBLE_BUFFER * HALF_TO_FP32), static_cast<uint32_t>(ALIGN_32));
        workspace.fp32ScaleTensor = LocalTensor<float>(TPosition::VECCALC, tempAddr,
            fp32ScaleBufAlign / sizeof(float));
    }
    return batchLength;
}

template <typename TopkWeightsType>
__aicore__ inline void LoadCombineReduceWeights(int32_t tileOffset, int32_t batchStart, int32_t tokenCount,
    const CombineReduceShapeContext& shapeContext, const CombineReduceBufferContext& bufferContext,
    CombineReduceWorkspace& workspace)
{
    if constexpr (Std::IsSame<TopkWeightsType, float>::value) {
        GlobalTensor<float> topKWeightsGlobal;
        topKWeightsGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(bufferContext.topKWeightsAddr));
        DataCopyExtParams copyParams{
            1U, static_cast<uint32_t>(tokenCount * shapeContext.topK * sizeof(float)), 0U, 0U, 0U};
        DataCopyPadExtParams<float> copyPadParams{false, 0U, 0U, 0U};
        DataCopyPad(workspace.topKWeightsTensor,
            topKWeightsGlobal[(tileOffset + batchStart) * shapeContext.topK], copyParams, copyPadParams);
        SetFlag<AscendC::HardEvent::MTE2_S>(0);
        WaitFlag<AscendC::HardEvent::MTE2_S>(0);
    }
    if constexpr (Std::IsSame<TopkWeightsType, bfloat16_t>::value) {
        GlobalTensor<bfloat16_t> topKWeightsGlobal;
        topKWeightsGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(bufferContext.topKWeightsAddr));
        DataCopyExtParams copyParams{
            1U, static_cast<uint32_t>(tokenCount * shapeContext.topK * sizeof(bfloat16_t)), 0U, 0U, 0U};
        DataCopyPadExtParams<bfloat16_t> copyPadParams{false, 0U, 0U, 0U};
        DataCopyPad(workspace.topKWeightsBf16Tensor,
            topKWeightsGlobal[(tileOffset + batchStart) * shapeContext.topK], copyParams, copyPadParams);
        SyncFuncStatic<AscendC::HardEvent::MTE2_V, SYNC_EVENT_ID2>();
        Cast(workspace.topKWeightsTensor, workspace.topKWeightsBf16Tensor,
            AscendC::RoundMode::CAST_NONE, tokenCount * shapeContext.topK);
        SetFlag<AscendC::HardEvent::V_S>(0);
        WaitFlag<AscendC::HardEvent::V_S>(0);
    }
}

template <int32_t CombineQuantMode>
__aicore__ inline void ProcessCombineReduceToken(int32_t tokenIdx, int32_t localIdx,
    const CombineReduceShapeContext& shapeContext, const GlobalTensor<bfloat16_t>& expandedX,
    CombineReduceWorkspace& workspace)
{
    LocalTensor<bfloat16_t> dataIn0Bf16 = workspace.dataResTensor[shapeContext.k];
    LocalTensor<bfloat16_t> dataIn1Bf16 = workspace.dataResTensor[shapeContext.k * 2];
    LocalTensor<float> dataIn0Fp32 = workspace.dataResFp32Tensor[shapeContext.k];
    LocalTensor<float> dataIn1Fp32 = workspace.dataResFp32Tensor[shapeContext.k * 2];
    for (int32_t expertId = 0; expertId < static_cast<int32_t>(shapeContext.topK); ++expertId) {
        float expertScale = workspace.topKWeightsTensor.GetValue(localIdx * shapeContext.topK + expertId);
        auto event = (expertId % DOUBLE_BUFFER == 0) ? EVENT_ID0 : EVENT_ID1;
        auto dataInBf16 = (expertId % DOUBLE_BUFFER == 0) ? dataIn0Bf16 : dataIn1Bf16;
        auto dataInFp32 = (expertId % DOUBLE_BUFFER == 0) ? dataIn0Fp32 : dataIn1Fp32;
        if constexpr (CombineQuantMode == COMBINE_NO_QUANT) {
            WaitFlag<AscendC::HardEvent::V_MTE2>(event);
            DataCopy(dataInBf16,
                expandedX[(static_cast<uint64_t>(tokenIdx) * shapeContext.topK + expertId) * shapeContext.k],
                shapeContext.k);
            SetFlag<AscendC::HardEvent::MTE2_V>(event);
            WaitFlag<AscendC::HardEvent::MTE2_V>(event);
            SetFlag<AscendC::HardEvent::S_V>(event);
            WaitFlag<AscendC::HardEvent::S_V>(event);
            Cast(dataInFp32, dataInBf16, AscendC::RoundMode::CAST_NONE, shapeContext.k);
        } else {
            uint32_t scaleNum = Ops::Base::CeilDiv(shapeContext.k, uint32_t(MXFP_SCALE_GROUP_NUM));
            uint32_t quantTokenSize = shapeContext.k + scaleNum;
            uint32_t quantElementNum = quantTokenSize / sizeof(bfloat16_t);
            WaitFlag<AscendC::HardEvent::V_MTE2>(event);
            DataCopy(dataInBf16,
                expandedX[(static_cast<uint64_t>(tokenIdx) * shapeContext.topK + expertId) * quantElementNum],
                quantElementNum);
            SetFlag<AscendC::HardEvent::MTE2_V>(event);
            WaitFlag<AscendC::HardEvent::MTE2_V>(event);
            using Fp8Type = typename std::conditional<CombineQuantMode == MXFP8_E4M3_COMM_QUANT,
                fp8_e4m3fn_t, fp8_e5m2_t>::type;
            MegaMoeCombineImpl::DeQuantMxFp8<Fp8Type, bfloat16_t>(dataInBf16, dataInFp32,
                workspace.bf16ScaleTensor, workspace.fp32ScaleTensor, scaleNum, shapeContext.k);
        }
        PipeBarrier<PIPE_V>();
        if (expertId == 0) {
            Muls(workspace.dataResFp32Tensor, dataInFp32, expertScale, shapeContext.k);
        } else {
            Muls(dataInFp32, dataInFp32, expertScale, shapeContext.k);
            PipeBarrier<PIPE_V>();
            Add(workspace.dataResFp32Tensor, workspace.dataResFp32Tensor, dataInFp32, shapeContext.k);
            PipeBarrier<PIPE_V>();
        }
        SetFlag<AscendC::HardEvent::V_MTE2>(event);
    }
}

template <int32_t CombineQuantMode, typename TopkWeightsType>
__aicore__ inline void ProcessCombineReduceSingleCore(const CombineReduceTileSchedule& tileSchedule,
    const CombineReduceShapeContext& shapeContext, const CombineReduceBufferContext& bufferContext)
{
    int32_t tileLength = 0;
    int32_t tileOffset = 0;
    ComputeCombineReduceTileRange(tileSchedule, shapeContext.m, tileLength, tileOffset);
    if (tileLength == 0) {
        return;
    }

    CombineReduceWorkspace workspace;
    int32_t batchLength = InitCombineReduceWorkspace<CombineQuantMode, TopkWeightsType>(
        tileLength, shapeContext, workspace);
    GlobalTensor<bfloat16_t> expandedX;
    expandedX.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(bufferContext.expandedXAddr));
    GlobalTensor<bfloat16_t> output;
    output.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(bufferContext.outputAddr));

    for (int32_t batchStart = 0; batchStart < tileLength; batchStart += batchLength) {
        int32_t tokenCount = (batchStart + batchLength > tileLength) ?
            (tileLength - batchStart) : batchLength;
        LoadCombineReduceWeights<TopkWeightsType>(
            tileOffset, batchStart, tokenCount, shapeContext, bufferContext, workspace);
        SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
        for (int32_t localIdx = 0; localIdx < tokenCount; ++localIdx) {
            int32_t tokenIdx = tileOffset + batchStart + localIdx;
            SyncFuncStatic<AscendC::HardEvent::MTE3_MTE2, SYNC_EVENT_ID2>();
            ProcessCombineReduceToken<CombineQuantMode>(tokenIdx, localIdx, shapeContext, expandedX, workspace);
            Cast(workspace.dataResTensor, workspace.dataResFp32Tensor, AscendC::RoundMode::CAST_RINT,
                shapeContext.k);
            SyncFuncStatic<AscendC::HardEvent::V_MTE3, SYNC_EVENT_ID3>();
            DataCopy(output[static_cast<uint64_t>(tokenIdx) * shapeContext.k], workspace.dataResTensor,
                shapeContext.k);
        }
        WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
    }
}

__aicore__ inline void CrossRankSyncSingleCore(const CombineReduceTileSchedule& tileSchedule,
    const CombineReduceSyncContext& syncContext)
{
    if constexpr (g_coreType == AIC) {
        return;
    }
    if (tileSchedule.tileNum == 0 || tileSchedule.tileId >= tileSchedule.tileNum) {
        return;
    }
    __gm__ int32_t* syncRank = reinterpret_cast<__gm__ int32_t*>(syncContext.rankSyncAddr);
    __gm__ int32_t* syncCount = reinterpret_cast<__gm__ int32_t*>(
        syncContext.rankSyncAddr + 48 * 1024 + tileSchedule.tileId * 64);
    int32_t count = ReadGmByPassDCache(syncCount) + 1;
    for (uint32_t rankIndex = tileSchedule.tileId; rankIndex < syncContext.worldSize;
        rankIndex += tileSchedule.tileNum) {
        __gm__ int32_t* syncRemoteAddr = reinterpret_cast<__gm__ int32_t*>(winRankAddr_[rankIndex])
            + syncContext.rankId * 16;
        WriteGmByPassDCache(syncRemoteAddr, count);
        GmSignalWaitBarrier(syncRank + rankIndex * 16, count);
    }
    WriteGmByPassDCache(syncCount, count);
    PipeBarrier<PIPE_ALL>();
    SyncAll<true>();
}

} // namespace MegaMoeCombineReduceImpl

#endif // MEGA_MOE_COMBINE_REDUCE_H
