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
 * \brief Pure AIV CombineReduce compute implementation.
 */

#ifndef MEGA_MOE_COMBINE_REDUCE_H
#define MEGA_MOE_COMBINE_REDUCE_H

#include "kernel_operator.h"
#include "../mega_moe_base.h"
#include "../mega_moe_tiling.h"
#include "mega_moe_combine_send_compute.h"

using namespace AscendC;

namespace MegaMoeCombineReduceImpl {
struct CombineReduceTileSchedule {
    uint32_t jobIndex;
    uint32_t totalJobs;
};

struct CombineReduceShapeContext {
    uint32_t m;
    uint32_t k;
    uint32_t topK;
    uint32_t sharedExpertNum;
};

struct CombineReduceBufferContext {
    GM_ADDR expandedXAddr;
    GM_ADDR topKWeightsAddr;
    GM_ADDR sharedExpertResultAddr;
    GM_ADDR outputAddr;
};

// CombineReduce 在 UB 中使用的数据、权重及反量化 scale buffer。
struct CombineReduceWorkspace {
    LocalTensor<bfloat16_t> dataResTensor;
    LocalTensor<float> dataResFp32Tensor;
    LocalTensor<float> topKWeightsTensor;
    LocalTensor<bfloat16_t> topKWeightsBf16Tensor;
    LocalTensor<bfloat16_t> bf16ScaleTensor;
    LocalTensor<float> fp32ScaleTensor;
};

// ===============================================================
// ComputeCombineReduceCoreRange：复用原 TilingByCore 公式，核 ID/核数改由逻辑核调度传入
// ===============================================================
__aicore__ inline void ComputeCombineReduceCoreRange(const CombineReduceTileSchedule& tileSchedule,
    uint32_t totalLength, int32_t& coreLen, int32_t& coreOffset)
{
    coreLen = 0;
    coreOffset = 0;
    if (tileSchedule.totalJobs == 0 || tileSchedule.jobIndex >= tileSchedule.totalJobs) {
        return;
    }

    int32_t totalLen = static_cast<int32_t>(totalLength);
    int32_t coreIdx = static_cast<int32_t>(tileSchedule.jobIndex);
    int32_t coreNum = static_cast<int32_t>(tileSchedule.totalJobs);
    int32_t lenPerCore = Ops::Base::CeilDiv(
        static_cast<uint32_t>(totalLen), static_cast<uint32_t>(coreNum));
    int32_t lenPerCoreAlign = Ops::Base::CeilAlign(
        static_cast<uint32_t>(lenPerCore), static_cast<uint32_t>(1));
    coreLen = lenPerCoreAlign;
    coreOffset = coreIdx * lenPerCoreAlign;
    if (coreOffset + coreLen >= totalLen) {
        coreLen = totalLen - coreOffset;
    }
    if (coreOffset >= totalLen) {
        coreLen = 0;
    }
}

// ===============================================================
// InitCombineReduceWorkspace：分配 CombineReduce 所需固定 buffer，返回 batchLength
// ===============================================================
template <int32_t CombineQuantMode, typename TopkWeightsType>
__aicore__ inline void InitCombineReduceWorkspace(const CombineReduceShapeContext& shapeContext,
    const MegaMoeUnpermuteBufferConfig& bufferConfig, CombineReduceWorkspace& workspace)
{
    uint32_t bf16ScaleBufAlign = 0;
    uint32_t fp32ScaleBufAlign = 0;
    if constexpr (CombineQuantMode != COMBINE_NO_QUANT) {
        uint32_t scaleNum = Ops::Base::CeilDiv(shapeContext.k, static_cast<uint32_t>(ALIGN_32));
        bf16ScaleBufAlign = Ops::Base::CeilAlign(
            static_cast<uint32_t>(scaleNum * sizeof(bfloat16_t) * DEQUANT_BF16_SCALE_EXPANSION),
            static_cast<uint32_t>(ALIGN_32));
        fp32ScaleBufAlign = Ops::Base::CeilAlign(
            static_cast<uint32_t>(scaleNum * sizeof(float) * DEQUANT_FP32_SCALE_EXPANSION),
            static_cast<uint32_t>(ALIGN_32));
    }

    uint32_t bf16SlotBytes = bufferConfig.bf16SlotElementCount * sizeof(bfloat16_t);
    uint32_t fp32SlotBytes = bufferConfig.fp32SlotElementCount * sizeof(float);
    uint32_t dataResBufAlign = (bufferConfig.inputBufferCount + 1) * bf16SlotBytes;
    uint32_t dataResFp32BufAlign = (bufferConfig.inputBufferCount + 1) * fp32SlotBytes;
    uint32_t dataResAddr = 0;
    workspace.dataResTensor = LocalTensor<bfloat16_t>(TPosition::VECCALC, dataResAddr,
        dataResBufAlign / sizeof(bfloat16_t));
    uint32_t dataResFp32Addr = dataResAddr + dataResBufAlign;
    workspace.dataResFp32Tensor = LocalTensor<float>(TPosition::VECCALC, dataResFp32Addr,
        dataResFp32BufAlign / sizeof(float));
    uint32_t tempAddr = dataResFp32Addr + dataResFp32BufAlign;

    workspace.topKWeightsTensor = LocalTensor<float>(TPosition::VECCALC, tempAddr,
        bufferConfig.topKWeightsBufferBytes / sizeof(float));
    tempAddr += bufferConfig.topKWeightsBufferBytes;

    if constexpr (Std::IsSame<TopkWeightsType, bfloat16_t>::value) {
        workspace.topKWeightsBf16Tensor = LocalTensor<bfloat16_t>(TPosition::VECCALC, tempAddr,
            bufferConfig.topKWeightsConversionBufferBytes / sizeof(bfloat16_t));
        tempAddr += bufferConfig.topKWeightsConversionBufferBytes;
    }

    if constexpr (CombineQuantMode != COMBINE_NO_QUANT) {
        workspace.bf16ScaleTensor = LocalTensor<bfloat16_t>(TPosition::VECCALC, tempAddr,
            bf16ScaleBufAlign / sizeof(bfloat16_t));
        tempAddr += bf16ScaleBufAlign;
        workspace.fp32ScaleTensor = LocalTensor<float>(TPosition::VECCALC, tempAddr,
            fp32ScaleBufAlign / sizeof(float));
    }
}

// ===============================================================
// LoadCombineReduceWeights：加载一个 token batch 的 topK 权重到 UB
// ===============================================================
template <typename TopkWeightsType>
__aicore__ inline void LoadCombineReduceWeights(int32_t coreOffset, int32_t batchStart, int32_t tokenCount,
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
            topKWeightsGlobal[(coreOffset + batchStart) * shapeContext.topK], copyParams, copyPadParams);
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
            topKWeightsGlobal[(coreOffset + batchStart) * shapeContext.topK], copyParams, copyPadParams);
        SyncFuncStatic<AscendC::HardEvent::MTE2_V, SYNC_EVENT_ID2>();
        Cast(workspace.topKWeightsTensor, workspace.topKWeightsBf16Tensor,
            AscendC::RoundMode::CAST_NONE, tokenCount * shapeContext.topK);
        SetFlag<AscendC::HardEvent::V_S>(0);
        WaitFlag<AscendC::HardEvent::V_S>(0);
    }
}

// ===============================================================
// ProcessCombineReduceToken：对单个 token 的 topK expert 结果做反量化、加权和累加
// ===============================================================
template <int32_t CombineQuantMode>
__aicore__ inline void ProcessCombineReduceToken(int32_t tokenIdx, int32_t localIdx,
    const CombineReduceShapeContext& shapeContext, const GlobalTensor<bfloat16_t>& expandedX,
    const MegaMoeUnpermuteBufferConfig& bufferConfig, CombineReduceWorkspace& workspace)
{
    for (int32_t expertId = 0; expertId < static_cast<int32_t>(shapeContext.topK); ++expertId) {
        float expertScale = workspace.topKWeightsTensor.GetValue(localIdx * shapeContext.topK + expertId);
        int32_t accumulationItemIdxInBatch =
            localIdx * (shapeContext.topK + shapeContext.sharedExpertNum) + expertId;
        int32_t inputBufferIdx = accumulationItemIdxInBatch % bufferConfig.inputBufferCount;
        TEventID event = static_cast<TEventID>(inputBufferIdx);
        LocalTensor<bfloat16_t> dataInBf16 =
            workspace.dataResTensor[(inputBufferIdx + 1) * bufferConfig.bf16SlotElementCount];
        LocalTensor<float> dataInFp32 =
            workspace.dataResFp32Tensor[(inputBufferIdx + 1) * bufferConfig.fp32SlotElementCount];
        if constexpr (CombineQuantMode == COMBINE_NO_QUANT) {
            WaitFlag<AscendC::HardEvent::V_MTE2>(event);
            DataCopy(dataInBf16,
                expandedX[(static_cast<uint64_t>(tokenIdx) * shapeContext.topK + expertId) * shapeContext.k],
                shapeContext.k);
            SetFlag<AscendC::HardEvent::MTE2_V>(event);
            WaitFlag<AscendC::HardEvent::MTE2_V>(event);
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
        SetFlag<AscendC::HardEvent::S_V>(event);
        WaitFlag<AscendC::HardEvent::S_V>(event);
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

__aicore__ inline void ProcessSharedExpertToken(int32_t tokenIdx, int32_t localIdx,
    const CombineReduceShapeContext& shapeContext, const GlobalTensor<bfloat16_t>& sharedExpertResult,
    const MegaMoeUnpermuteBufferConfig& bufferConfig, CombineReduceWorkspace& workspace)
{
    for (uint32_t sharedIdx = 0; sharedIdx < shapeContext.sharedExpertNum; ++sharedIdx) {
        int32_t accumulationItemIdxInBatch =
            localIdx * (shapeContext.topK + shapeContext.sharedExpertNum) + shapeContext.topK + sharedIdx;
        int32_t inputBufferIdx = accumulationItemIdxInBatch % bufferConfig.inputBufferCount;
        TEventID event = static_cast<TEventID>(inputBufferIdx);
        LocalTensor<bfloat16_t> dataInBf16 =
            workspace.dataResTensor[(inputBufferIdx + 1) * bufferConfig.bf16SlotElementCount];
        LocalTensor<float> dataInFp32 =
            workspace.dataResFp32Tensor[(inputBufferIdx + 1) * bufferConfig.fp32SlotElementCount];
        WaitFlag<AscendC::HardEvent::V_MTE2>(event);
        DataCopy(dataInBf16,
            sharedExpertResult[(static_cast<uint64_t>(sharedIdx) * shapeContext.m + tokenIdx) * shapeContext.k],
            shapeContext.k);
        SetFlag<AscendC::HardEvent::MTE2_V>(event);
        WaitFlag<AscendC::HardEvent::MTE2_V>(event);
        SetFlag<AscendC::HardEvent::S_V>(event);
        WaitFlag<AscendC::HardEvent::S_V>(event);
        Cast(dataInFp32, dataInBf16, AscendC::RoundMode::CAST_NONE, shapeContext.k);
        PipeBarrier<PIPE_V>();
        Add(workspace.dataResFp32Tensor, workspace.dataResFp32Tensor, dataInFp32, shapeContext.k);
        PipeBarrier<PIPE_V>();
        SetFlag<AscendC::HardEvent::V_MTE2>(event);
    }
}

// ===============================================================
// ProcessCombineReduceSingleCore：单核入口——分核、初始化 buffer，再按 batch 执行 CombineReduce
// ===============================================================
template <int32_t CombineQuantMode, typename TopkWeightsType>
__aicore__ inline void ProcessCombineReduceSingleCore(const CombineReduceTileSchedule& tileSchedule,
    const CombineReduceShapeContext& shapeContext, const CombineReduceBufferContext& bufferContext,
    const MegaMoeUnpermuteBufferConfig& bufferConfig)
{
    int32_t coreLen = 0;
    int32_t coreOffset = 0;
    ComputeCombineReduceCoreRange(tileSchedule, shapeContext.m, coreLen, coreOffset);
    if (coreLen == 0) {
        return;
    }

    CombineReduceWorkspace workspace;
    InitCombineReduceWorkspace<CombineQuantMode, TopkWeightsType>(shapeContext, bufferConfig, workspace);
    int32_t batchLength = bufferConfig.tokensPerBatch;
    GlobalTensor<bfloat16_t> expandedX;
    expandedX.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(bufferContext.expandedXAddr));
    GlobalTensor<bfloat16_t> sharedExpertResult;
    if (shapeContext.sharedExpertNum > 0) {
        sharedExpertResult.SetGlobalBuffer(
            reinterpret_cast<__gm__ bfloat16_t*>(bufferContext.sharedExpertResultAddr));
    }
    GlobalTensor<bfloat16_t> output;
    output.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(bufferContext.outputAddr));

    constexpr TEventID outputBufferEvent = EVENT_ID0;
    SetFlag<AscendC::HardEvent::MTE3_V>(outputBufferEvent);
    for (int32_t batchStart = 0; batchStart < coreLen; batchStart += batchLength) {
        int32_t tokenCount = (batchStart + batchLength > coreLen) ?
            (coreLen - batchStart) : batchLength;
        LoadCombineReduceWeights<TopkWeightsType>(
            coreOffset, batchStart, tokenCount, shapeContext, bufferContext, workspace);
        for (int32_t bufferIdx = 0; bufferIdx < bufferConfig.inputBufferCount; ++bufferIdx) {
            SetFlag<AscendC::HardEvent::V_MTE2>(static_cast<TEventID>(bufferIdx));
        }
        for (int32_t localIdx = 0; localIdx < tokenCount; ++localIdx) {
            int32_t tokenIdx = coreOffset + batchStart + localIdx;
            ProcessCombineReduceToken<CombineQuantMode>(
                tokenIdx, localIdx, shapeContext, expandedX, bufferConfig, workspace);
            if (shapeContext.sharedExpertNum > 0) {
                ProcessSharedExpertToken(
                    tokenIdx, localIdx, shapeContext, sharedExpertResult, bufferConfig, workspace);
            }
            WaitFlag<AscendC::HardEvent::MTE3_V>(outputBufferEvent);
            Cast(workspace.dataResTensor, workspace.dataResFp32Tensor, AscendC::RoundMode::CAST_RINT,
                shapeContext.k);
            SyncFuncStatic<AscendC::HardEvent::V_MTE3, SYNC_EVENT_ID3>();
            DataCopy(output[static_cast<uint64_t>(tokenIdx) * shapeContext.k], workspace.dataResTensor,
                shapeContext.k);
            SetFlag<AscendC::HardEvent::MTE3_V>(outputBufferEvent);
        }
        for (int32_t bufferIdx = 0; bufferIdx < bufferConfig.inputBufferCount; ++bufferIdx) {
            WaitFlag<AscendC::HardEvent::V_MTE2>(static_cast<TEventID>(bufferIdx));
        }
    }
    WaitFlag<AscendC::HardEvent::MTE3_V>(outputBufferEvent);
}

} // namespace MegaMoeCombineReduceImpl

#endif // MEGA_MOE_COMBINE_REDUCE_H
