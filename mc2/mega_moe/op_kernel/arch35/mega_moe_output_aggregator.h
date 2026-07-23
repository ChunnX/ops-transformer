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
 * \file mega_moe_output_aggregator.h
 * \brief Final token output aggregation for MegaMoe.
 */

#ifndef MEGA_MOE_OUTPUT_AGGREGATOR_H
#define MEGA_MOE_OUTPUT_AGGREGATOR_H

#include "kernel_operator.h"
#include "mega_moe_base.h"
#include "mega_moe_combine_send.h"

namespace MegaMoeImpl {
using namespace AscendC;

struct OutputAggregationTensorAddresses {
    GM_ADDR topKWeightsGmAddr;
    GM_ADDR routedExpertOutputGmAddr;
    GM_ADDR sharedExpertOutputGmAddr;
    GM_ADDR outputGmAddr;
};

struct OutputAggregationDimensions {
    uint32_t tokenCount;
    uint32_t hiddenSize;
    uint32_t topK;
    uint32_t sharedExpertCount;
};

template <typename TopkWeightsType, int32_t CombineQuantMode>
class MegaMoeOutputAggregator {
public:
    __aicore__ inline MegaMoeOutputAggregator(const OutputAggregationTensorAddresses &tensorAddresses,
                                              const OutputAggregationDimensions &dimensions,
                                              const MegaMoeUnpermuteBufferConfig &bufferConfig)
        : tensorAddresses_(tensorAddresses), dimensions_(dimensions), bufferConfig_(bufferConfig)
    {
    }

    __aicore__ inline void Aggregate();

private:
    using BufferConfig = MegaMoeUnpermuteBufferConfig;

    template <bool IsShared>
    __aicore__ inline void AggregateImpl(int32_t coreLen, int32_t coreOffset);
    __aicore__ inline void InitBuffers();
    __aicore__ inline void LoadTopKWeights(int32_t coreOffset, int32_t batchTokenOffset, int32_t batchTokenCount);
    template <bool IsShared>
    __aicore__ inline void AccumulateRoutedExpertOutputs(int32_t tokenIdx, int32_t localIdx,
                                                         const GlobalTensor<bfloat16_t> &routedExpertOutput);
    __aicore__ inline void AccumulateSharedExpertOutputs(int32_t tokenIdx, int32_t localIdx);

    OutputAggregationTensorAddresses tensorAddresses_;
    OutputAggregationDimensions dimensions_;
    BufferConfig bufferConfig_;
    LocalTensor<bfloat16_t> dataResTensor_;
    LocalTensor<float> dataResFp32Tensor_;
    LocalTensor<float> topKWeightsTensor_;
    LocalTensor<float> fp32ScaleTensor_;
    LocalTensor<bfloat16_t> bf16ScaleTensor_;
    LocalTensor<bfloat16_t> topKWeightsBf16Tensor_;
};

template <typename TopkWeightsType, int32_t CombineQuantMode>
__aicore__ inline void MegaMoeOutputAggregator<TopkWeightsType, CombineQuantMode>::InitBuffers()
{
    uint32_t bf16ScaleBufAlign = 0;
    uint32_t fp32ScaleBufAlign = 0;
    if constexpr (CombineQuantMode != COMBINE_NO_QUANT) {
        uint32_t scaleNum = Ops::Base::CeilDiv(dimensions_.hiddenSize, static_cast<uint32_t>(ALIGN_32));
        bf16ScaleBufAlign =
            Ops::Base::CeilAlign(static_cast<uint32_t>(scaleNum * sizeof(bfloat16_t) * DEQUANT_BF16_SCALE_EXPANSION),
                                 static_cast<uint32_t>(ALIGN_32));
        fp32ScaleBufAlign =
            Ops::Base::CeilAlign(static_cast<uint32_t>(scaleNum * sizeof(float) * DEQUANT_FP32_SCALE_EXPANSION),
                                 static_cast<uint32_t>(ALIGN_32));
    }

    uint32_t bf16SlotBytes = bufferConfig_.bf16SlotElementCount * sizeof(bfloat16_t);
    uint32_t fp32SlotBytes = bufferConfig_.fp32SlotElementCount * sizeof(float);
    uint32_t topKWeightsBufAlign = bufferConfig_.topKWeightsBufferBytes;
    uint32_t topKWeightsConversionBufferBytes = bufferConfig_.topKWeightsConversionBufferBytes;

    uint32_t dataResBufAlign = (bufferConfig_.inputBufferCount + 1) * bf16SlotBytes;
    uint32_t dataResFp32BufAlign = (bufferConfig_.inputBufferCount + 1) * fp32SlotBytes;
    // Tensor用处：存储 MTE2 搬入 token；
    // Tensor大小：(1 + bufferConfig_.inputBufferCount) × 独立对齐后的 BF16 单槽大小；
    // 1 块用于累加/搬出，其余用于 MTE2 搬入。
    uint32_t dataResAddr = 0;
    dataResTensor_ = LocalTensor<bfloat16_t>(TPosition::VECCALC, dataResAddr, dataResBufAlign / sizeof(bfloat16_t));
    // Tensor用处：存储 token Cast 目的 Tensor；
    // Tensor大小：(1 + bufferConfig_.inputBufferCount) × 独立对齐后的 FP32 单槽大小；
    uint32_t dataResFp32Addr = dataResAddr + dataResBufAlign;
    dataResFp32Tensor_ = LocalTensor<float>(TPosition::VECCALC, dataResFp32Addr, dataResFp32BufAlign / sizeof(float));
    uint32_t tempAddr = dataResFp32Addr + dataResFp32BufAlign;

    // weight buffer（在 scale 之前，与原布局顺序一致）
    // Tensor用处：存储 topK weight；
    // Tensor大小：tokensPerBatch × topK × sizeof(float) align 到 32 字节对齐；
    topKWeightsTensor_ = LocalTensor<float>(TPosition::VECCALC, tempAddr, topKWeightsBufAlign / sizeof(float));
    tempAddr += topKWeightsBufAlign;

    if constexpr (Std::IsSame<TopkWeightsType, bfloat16_t>::value) {
        // Tensor用处：BF16 weight 搬运中转 buffer；
        // Tensor大小：tokensPerBatch × topK × sizeof(bfloat16_t) align 到 32 字节；
        topKWeightsBf16Tensor_ = LocalTensor<bfloat16_t>(TPosition::VECCALC, tempAddr,
                                                         topKWeightsConversionBufferBytes / sizeof(bfloat16_t));
        tempAddr += topKWeightsConversionBufferBytes;
    }

    if constexpr (CombineQuantMode != COMBINE_NO_QUANT) {
        // Tensor用处：DeQuantMxFp8 中用于存储 BF16 格式的 scale（E8M0 转换后的中间结果）。
        bf16ScaleTensor_ =
            LocalTensor<bfloat16_t>(TPosition::VECCALC, tempAddr, bf16ScaleBufAlign / sizeof(bfloat16_t));
        tempAddr += bf16ScaleBufAlign;
        // Tensor用处：DeQuantMxFp8 中用于存储 FP32 格式的 scale（广播后的最终 scale）。
        fp32ScaleTensor_ = LocalTensor<float>(TPosition::VECCALC, tempAddr, fp32ScaleBufAlign / sizeof(float));
        tempAddr += fp32ScaleBufAlign;
    }
}

template <typename TopkWeightsType, int32_t CombineQuantMode>
__aicore__ inline void MegaMoeOutputAggregator<TopkWeightsType, CombineQuantMode>::LoadTopKWeights(
    int32_t coreOffset, int32_t batchTokenOffset, int32_t batchTokenCount)
{
    if constexpr (Std::IsSame<TopkWeightsType, float>::value) {
        GlobalTensor<float> topKWeightsGlobalTensor_;
        topKWeightsGlobalTensor_.SetGlobalBuffer((__gm__ float *)tensorAddresses_.topKWeightsGmAddr);
        DataCopyExtParams copyParams = {1U, static_cast<uint32_t>(batchTokenCount * dimensions_.topK * sizeof(float)),
                                        0U, 0U, 0U};
        DataCopyPadExtParams<float> copyPadParams{false, 0U, 0U, 0U};
        DataCopyPad(topKWeightsTensor_, topKWeightsGlobalTensor_[(coreOffset + batchTokenOffset) * dimensions_.topK],
                    copyParams, copyPadParams);
        SetFlag<AscendC::HardEvent::MTE2_S>(0);
        WaitFlag<AscendC::HardEvent::MTE2_S>(0);
    }
    if constexpr (Std::IsSame<TopkWeightsType, bfloat16_t>::value) {
        GlobalTensor<bfloat16_t> topkWeightsGlobalTensor;
        topkWeightsGlobalTensor.SetGlobalBuffer((__gm__ bfloat16_t *)tensorAddresses_.topKWeightsGmAddr);
        DataCopyExtParams copyParams = {
            1U, static_cast<uint32_t>(batchTokenCount * dimensions_.topK * sizeof(bfloat16_t)), 0U, 0U, 0U};
        DataCopyPadExtParams<bfloat16_t> copyPadParams{false, 0U, 0U, 0U};
        DataCopyPad(topKWeightsBf16Tensor_, topkWeightsGlobalTensor[(coreOffset + batchTokenOffset) * dimensions_.topK],
                    copyParams, copyPadParams);
        SyncFuncStatic<AscendC::HardEvent::MTE2_V, SYNC_EVENT_ID2>();
        Cast(topKWeightsTensor_, topKWeightsBf16Tensor_, AscendC::RoundMode::CAST_NONE,
             batchTokenCount * dimensions_.topK);
        SetFlag<AscendC::HardEvent::V_S>(0);
        WaitFlag<AscendC::HardEvent::V_S>(0);
    }
}

template <typename TopkWeightsType, int32_t CombineQuantMode>
template <bool IsShared>
__aicore__ inline void MegaMoeOutputAggregator<TopkWeightsType, CombineQuantMode>::AccumulateRoutedExpertOutputs(
    int32_t tokenIdx, int32_t localIdx, const GlobalTensor<bfloat16_t> &expandedX)
{
    for (int32_t expId = 0; expId < dimensions_.topK; ++expId) {
        float expScale = topKWeightsTensor_.GetValue(localIdx * dimensions_.topK + expId);
        // Routed and shared expert results form one continuous accumulation-input sequence in the dynamic ring.
        int32_t accumulationItemCountPerToken = dimensions_.topK;
        if constexpr (IsShared) {
            accumulationItemCountPerToken += static_cast<int32_t>(dimensions_.sharedExpertCount);
        }
        int32_t accumulationItemIdxInBatch = localIdx * accumulationItemCountPerToken + expId;
        int32_t inputBufferIdx = accumulationItemIdxInBatch % bufferConfig_.inputBufferCount;
        TEventID event = static_cast<TEventID>(inputBufferIdx);
        LocalTensor<bfloat16_t> dataInBf16 = dataResTensor_[(inputBufferIdx + 1) * bufferConfig_.bf16SlotElementCount];
        LocalTensor<float> dataInFp32 = dataResFp32Tensor_[(inputBufferIdx + 1) * bufferConfig_.fp32SlotElementCount];
        if constexpr (CombineQuantMode == COMBINE_NO_QUANT) {
            WaitFlag<AscendC::HardEvent::V_MTE2>(event);
            DataCopy(dataInBf16,
                     expandedX[(static_cast<uint64_t>(tokenIdx) * dimensions_.topK + expId) * dimensions_.hiddenSize],
                     dimensions_.hiddenSize);
            SetFlag<AscendC::HardEvent::MTE2_V>(event);
            WaitFlag<AscendC::HardEvent::MTE2_V>(event);
            Cast(dataInFp32, dataInBf16, AscendC::RoundMode::CAST_NONE, dimensions_.hiddenSize);
        } else {
            uint32_t nScale = Ops::Base::CeilDiv(dimensions_.hiddenSize, uint32_t(MXFP_SCALE_GROUP_NUM));
            uint32_t quantTokenSize = dimensions_.hiddenSize + nScale;
            uint32_t quantEleNum = quantTokenSize / sizeof(bfloat16_t);
            WaitFlag<AscendC::HardEvent::V_MTE2>(event);
            DataCopy(dataInBf16, expandedX[(static_cast<uint64_t>(tokenIdx) * dimensions_.topK + expId) * quantEleNum],
                     quantEleNum);
            SetFlag<AscendC::HardEvent::MTE2_V>(event);
            WaitFlag<AscendC::HardEvent::MTE2_V>(event);
            using Fp8Type =
                typename std::conditional<CombineQuantMode == MXFP8_E4M3_COMM_QUANT, fp8_e4m3fn_t, fp8_e5m2_t>::type;
            MegaMoeCombineImpl::DeQuantMxFp8<Fp8Type, bfloat16_t>(dataInBf16, dataInFp32, bf16ScaleTensor_,
                                                                  fp32ScaleTensor_, nScale, dimensions_.hiddenSize);
        }
        // GetValue 在 Scalar 流水读取 expScale；两条反量化路径汇合后统一等待，再由 Vector 流水消费。
        SetFlag<AscendC::HardEvent::S_V>(event);
        WaitFlag<AscendC::HardEvent::S_V>(event);
        PipeBarrier<PIPE_V>();
        if (expId == 0) {
            Muls(dataResFp32Tensor_, dataInFp32, expScale, dimensions_.hiddenSize);
        } else {
            Muls(dataInFp32, dataInFp32, expScale, dimensions_.hiddenSize);
            PipeBarrier<PIPE_V>();
            Add(dataResFp32Tensor_, dataResFp32Tensor_, dataInFp32, dimensions_.hiddenSize);
            PipeBarrier<PIPE_V>();
        }
        SetFlag<AscendC::HardEvent::V_MTE2>(event);
    }
}

template <typename TopkWeightsType, int32_t CombineQuantMode>
__aicore__ inline void
MegaMoeOutputAggregator<TopkWeightsType, CombineQuantMode>::AccumulateSharedExpertOutputs(int32_t tokenIdx,
                                                                                          int32_t localIdx)
{
    GlobalTensor<bfloat16_t> sharedResult;
    sharedResult.SetGlobalBuffer((__gm__ bfloat16_t *)tensorAddresses_.sharedExpertOutputGmAddr);
    for (uint32_t sharedIdx = 0; sharedIdx < dimensions_.sharedExpertCount; sharedIdx++) {
        int32_t accumulationItemIdxInBatch =
            localIdx * (dimensions_.topK + static_cast<int32_t>(dimensions_.sharedExpertCount)) + dimensions_.topK +
            static_cast<int32_t>(sharedIdx);
        int32_t inputBufferIdx = accumulationItemIdxInBatch % bufferConfig_.inputBufferCount;
        TEventID event = static_cast<TEventID>(inputBufferIdx);
        LocalTensor<bfloat16_t> dataInBf16 = dataResTensor_[(inputBufferIdx + 1) * bufferConfig_.bf16SlotElementCount];
        LocalTensor<float> dataInFp32 = dataResFp32Tensor_[(inputBufferIdx + 1) * bufferConfig_.fp32SlotElementCount];
        WaitFlag<AscendC::HardEvent::V_MTE2>(event);
        DataCopy(dataInBf16, sharedResult[(sharedIdx * dimensions_.tokenCount + tokenIdx) * dimensions_.hiddenSize],
                 dimensions_.hiddenSize);
        SetFlag<AscendC::HardEvent::MTE2_V>(event);
        WaitFlag<AscendC::HardEvent::MTE2_V>(event);
        SetFlag<AscendC::HardEvent::S_V>(event);
        WaitFlag<AscendC::HardEvent::S_V>(event);
        Cast(dataInFp32, dataInBf16, AscendC::RoundMode::CAST_NONE, dimensions_.hiddenSize);
        PipeBarrier<PIPE_V>();
        Add(dataResFp32Tensor_, dataResFp32Tensor_, dataInFp32, dimensions_.hiddenSize);
        PipeBarrier<PIPE_V>();
        SetFlag<AscendC::HardEvent::V_MTE2>(event);
    }
}

template <typename TopkWeightsType, int32_t CombineQuantMode>
__aicore__ inline void MegaMoeOutputAggregator<TopkWeightsType, CombineQuantMode>::Aggregate()
{
    int32_t coreLen, coreOffset;
    TilingByCore(dimensions_.tokenCount, coreLen, coreOffset, 1);
    if (coreLen == 0) {
        return;
    }
    InitBuffers();

    if (dimensions_.sharedExpertCount > 0) {
        AggregateImpl<true>(coreLen, coreOffset);
    } else {
        AggregateImpl<false>(coreLen, coreOffset);
    }
}

template <typename TopkWeightsType, int32_t CombineQuantMode>
template <bool IsShared>
__aicore__ inline void MegaMoeOutputAggregator<TopkWeightsType, CombineQuantMode>::AggregateImpl(int32_t coreLen,
                                                                                                 int32_t coreOffset)
{
    GlobalTensor<bfloat16_t> expandedX;
    expandedX.SetGlobalBuffer((__gm__ bfloat16_t *)tensorAddresses_.routedExpertOutputGmAddr);
    GlobalTensor<bfloat16_t> output;
    output.SetGlobalBuffer((__gm__ bfloat16_t *)tensorAddresses_.outputGmAddr);

    // 输出槽由 Vector 写入、MTE3 读出。先将槽位交给 Vector，最终回收最后一次 MTE3 完成信号。
    constexpr TEventID kOutputBufferEvent = EVENT_ID0;
    SetFlag<AscendC::HardEvent::MTE3_V>(kOutputBufferEvent);

    for (int32_t batchTokenOffset = 0; batchTokenOffset < coreLen; batchTokenOffset += bufferConfig_.tokensPerBatch) {
        int32_t batchTokenCount = (batchTokenOffset + bufferConfig_.tokensPerBatch > coreLen) ?
                                      (coreLen - batchTokenOffset) :
                                      bufferConfig_.tokensPerBatch;

        LoadTopKWeights(coreOffset, batchTokenOffset, batchTokenCount);

        for (int32_t bufferIdx = 0; bufferIdx < bufferConfig_.inputBufferCount; ++bufferIdx) {
            SetFlag<AscendC::HardEvent::V_MTE2>(static_cast<TEventID>(bufferIdx));
        }
        for (int32_t localIdx = 0; localIdx < batchTokenCount; localIdx++) {
            int32_t tokenIdx = coreOffset + batchTokenOffset + localIdx;
            AccumulateRoutedExpertOutputs<IsShared>(tokenIdx, localIdx, expandedX);
            if constexpr (IsShared) {
                AccumulateSharedExpertOutputs(tokenIdx, localIdx);
            }
            // MTE2 使用独立输入槽，可与上一 token 的 MTE3 输出重叠；仅在覆盖输出槽前等待 MTE3 读完。
            WaitFlag<AscendC::HardEvent::MTE3_V>(kOutputBufferEvent);
            Cast(dataResTensor_, dataResFp32Tensor_, AscendC::RoundMode::CAST_RINT, dimensions_.hiddenSize);
            SyncFuncStatic<AscendC::HardEvent::V_MTE3, SYNC_EVENT_ID3>();
            DataCopy(output[static_cast<uint64_t>(tokenIdx) * dimensions_.hiddenSize], dataResTensor_,
                     dimensions_.hiddenSize);
            SetFlag<AscendC::HardEvent::MTE3_V>(kOutputBufferEvent);
        }
        for (int32_t bufferIdx = 0; bufferIdx < bufferConfig_.inputBufferCount; ++bufferIdx) {
            WaitFlag<AscendC::HardEvent::V_MTE2>(static_cast<TEventID>(bufferIdx));
        }
    }
    WaitFlag<AscendC::HardEvent::MTE3_V>(kOutputBufferEvent);
}

} // namespace MegaMoeImpl

#endif // MEGA_MOE_OUTPUT_AGGREGATOR_H
