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
 * \brief MegaMoe CombineReduce member-function definitions.
 */

#ifndef MEGA_MOE_COMBINE_REDUCE_H
#define MEGA_MOE_COMBINE_REDUCE_H

// ===============================================================
// UnpermuteLoadWeights：加载一个 token batch 的权重到 UB
// ===============================================================
template <TemplateMegaMoeTypeClass>
__aicore__ inline void
MegaMoe<TemplateMegaMoeTypeFunc>::UnpermuteLoadWeights(int32_t coreOffset, int32_t batchTokenOffset,
                                                       int32_t batchTokenCount, LocalTensor<bfloat16_t> &tempLocal)
{
    if constexpr (Std::IsSame<TopkWeightsType, float>::value) {
        GlobalTensor<float> topKWeightsGlobalTensor_;
        topKWeightsGlobalTensor_.SetGlobalBuffer((__gm__ float *)params_.probsGmAddr);
        DataCopyExtParams copyParams = {1U, static_cast<uint32_t>(batchTokenCount * topK_ * sizeof(float)), 0U, 0U, 0U};
        DataCopyPadExtParams<float> copyPadParams{false, 0U, 0U, 0U};
        DataCopyPad(topKWeightsTensor_, topKWeightsGlobalTensor_[(coreOffset + batchTokenOffset) * topK_], copyParams,
                    copyPadParams);
        SetFlag<AscendC::HardEvent::MTE2_S>(0);
        WaitFlag<AscendC::HardEvent::MTE2_S>(0);
    }
    if constexpr (Std::IsSame<TopkWeightsType, bfloat16_t>::value) {
        GlobalTensor<bfloat16_t> topkWeightsGlobalTensor;
        topkWeightsGlobalTensor.SetGlobalBuffer((__gm__ bfloat16_t *)params_.probsGmAddr);
        DataCopyExtParams copyParams = {1U, static_cast<uint32_t>(batchTokenCount * topK_ * sizeof(bfloat16_t)), 0U, 0U,
                                        0U};
        DataCopyPadExtParams<bfloat16_t> copyPadParams{false, 0U, 0U, 0U};
        DataCopyPad(tempLocal, topkWeightsGlobalTensor[(coreOffset + batchTokenOffset) * topK_], copyParams,
                    copyPadParams);
        SyncFuncStatic<AscendC::HardEvent::MTE2_V, SYNC_EVENT_ID2>();
        Cast(topKWeightsTensor_, tempLocal, AscendC::RoundMode::CAST_NONE, batchTokenCount * topK_);
        SetFlag<AscendC::HardEvent::V_S>(0);
        WaitFlag<AscendC::HardEvent::V_S>(0);
    }
}

// ===============================================================
// UnpermuteProcessToken：单个 token 的 per-expert 累加
// ===============================================================
template <TemplateMegaMoeTypeClass>
__aicore__ inline void
MegaMoe<TemplateMegaMoeTypeFunc>::UnpermuteProcessToken(int32_t tokenIdx, int32_t localIdx,
                                                        const GlobalTensor<bfloat16_t> &expandedX)
{
    LocalTensor<bfloat16_t> dataIn0Bf16 = dataResTensor_[k_];
    LocalTensor<bfloat16_t> dataIn1Bf16 = dataResTensor_[k_ * 2];
    LocalTensor<float> dataIn0Fp32 = dataResFp32Tensor_[k_];
    LocalTensor<float> dataIn1Fp32 = dataResFp32Tensor_[k_ * 2];
    for (int32_t expId = 0; expId < topK_; ++expId) {
        float expScale = topKWeightsTensor_.GetValue(localIdx * topK_ + expId);
        auto event = (expId % DOUBLE_BUFFER == 0) ? EVENT_ID0 : EVENT_ID1;
        auto dataInBf16 = (expId % DOUBLE_BUFFER == 0) ? dataIn0Bf16 : dataIn1Bf16;
        auto dataInFp32 = (expId % DOUBLE_BUFFER == 0) ? dataIn0Fp32 : dataIn1Fp32;
        if constexpr (CombineQuantMode == COMBINE_NO_QUANT) {
            WaitFlag<AscendC::HardEvent::V_MTE2>(event);
            DataCopy(dataInBf16, expandedX[(static_cast<uint64_t>(tokenIdx) * topK_ + expId) * k_], k_);
            SetFlag<AscendC::HardEvent::MTE2_V>(event);
            WaitFlag<AscendC::HardEvent::MTE2_V>(event);
            SetFlag<AscendC::HardEvent::S_V>(event);
            WaitFlag<AscendC::HardEvent::S_V>(event);
            Cast(dataInFp32, dataInBf16, AscendC::RoundMode::CAST_NONE, k_);
        } else {
            uint32_t nScale = Ops::Base::CeilDiv(k_, uint32_t(MXFP_SCALE_GROUP_NUM));
            uint32_t quantTokenSize = k_ + nScale;
            uint32_t quantEleNum = quantTokenSize / sizeof(bfloat16_t);
            WaitFlag<AscendC::HardEvent::V_MTE2>(event);
            DataCopy(dataInBf16, expandedX[(static_cast<uint64_t>(tokenIdx) * topK_ + expId) * quantEleNum],
                     quantEleNum);
            SetFlag<AscendC::HardEvent::MTE2_V>(event);
            WaitFlag<AscendC::HardEvent::MTE2_V>(event);
            using Fp8Type =
                typename std::conditional<CombineQuantMode == MXFP8_E4M3_COMM_QUANT, fp8_e4m3fn_t, fp8_e5m2_t>::type;
            MegaMoeCombineImpl::DeQuantMxFp8<Fp8Type, bfloat16_t>(dataInBf16, dataInFp32, bf16ScaleTensor_,
                                                                  fp32ScaleTensor_, nScale, k_);
        }
        PipeBarrier<PIPE_V>();
        if (expId == 0) {
            Muls(dataResFp32Tensor_, dataInFp32, expScale, k_);
        } else {
            Muls(dataInFp32, dataInFp32, expScale, k_);
            PipeBarrier<PIPE_V>();
            Add(dataResFp32Tensor_, dataResFp32Tensor_, dataInFp32, k_);
            PipeBarrier<PIPE_V>();
        }
        SetFlag<AscendC::HardEvent::V_MTE2>(event);
    }
}

// ===============================================================
// UnpermuteBuffInit：分配 Unpermute 所需固定 buffer，返回每个 batch 处理的 token 数
// ===============================================================
template <TemplateMegaMoeTypeClass>
__aicore__ inline int32_t MegaMoe<TemplateMegaMoeTypeFunc>::UnpermuteBuffInit(int32_t coreLen)
{
    // 固定 buffer：dataRes + dataResFp32 (+ scales)
    uint32_t dataResBufAlign = Ops::Base::CeilAlign(static_cast<uint32_t>(UNPERMUTE_LIST_NUM * k_ * sizeof(bfloat16_t)),
                                                    static_cast<uint32_t>(ALIGN_32));
    uint32_t dataResFp32BufAlign = dataResBufAlign * HALF_TO_FP32;
    // Tensor用处：Unpermute 函数用于存储 mte2 搬入 token；
    // Tensor大小：3 * 单个 token 长度，2 块用于 mte2 搬运 doubleBuffer，1 块存储 Cast 输出结果用于搬出；
    uint32_t dataResAddr = 0;
    dataResTensor_ = LocalTensor<bfloat16_t>(TPosition::VECCALC, dataResAddr, dataResBufAlign / sizeof(bfloat16_t));
    // Tensor用处：Unpermute 函数用于存储 token Cast 目的 Tensor；
    // Tensor大小：dataResTensor_ 开设大小 × HALF_TO_FP32；
    uint32_t dataResFp32Addr = dataResAddr + dataResBufAlign;
    dataResFp32Tensor_ = LocalTensor<float>(TPosition::VECCALC, dataResFp32Addr, dataResFp32BufAlign / sizeof(float));
    uint32_t tempAddr = dataResFp32Addr + dataResFp32BufAlign;

    // 以 UNPERMUTE_BASE_TOPK 为基准确定每个 batch 处理的 token 数。
    int32_t tokensPerBatch = UNPERMUTE_BASE_TOKENS_PER_BATCH;
    if (topK_ > UNPERMUTE_BASE_TOPK) {
        // 保持 tokensPerBatch * topK_ 不超过基准 topK 时的 UB 预算。
        tokensPerBatch = tokensPerBatch * UNPERMUTE_BASE_TOPK / topK_;
    }
    if (tokensPerBatch > coreLen) {
        tokensPerBatch = coreLen;
    }

    // weight buffer（在 scale 之前，与 master 顺序一致）
    // Tensor用处：用于存储 topKWeight；
    // Tensor大小：tokensPerBatch × topK_ × sizeof(float) align 到 32 字节对齐；
    uint32_t topKWeightsBufAlign = Ops::Base::CeilAlign(static_cast<uint32_t>(tokensPerBatch * topK_ * sizeof(float)),
                                                        static_cast<uint32_t>(ALIGN_32));
    topKWeightsTensor_ = LocalTensor<float>(TPosition::VECCALC, tempAddr, topKWeightsBufAlign / sizeof(float));
    tempAddr += topKWeightsBufAlign;

    if constexpr (Std::IsSame<TopkWeightsType, bfloat16_t>::value) {
        // Tensor用处：Unpermute 中 bf16 weight 搬运中转 buffer；
        // Tensor大小：tokensPerBatch × topK_ × sizeof(bfloat16_t) align 到 32 字节；
        uint32_t tempBufAlign = Ops::Base::CeilAlign(static_cast<uint32_t>(tokensPerBatch * topK_ * sizeof(bfloat16_t)),
                                                     uint32_t(ALIGN_32));
        topKWeightsBf16Tensor_ =
            LocalTensor<bfloat16_t>(TPosition::VECCALC, tempAddr, tempBufAlign / sizeof(bfloat16_t));
        tempAddr += tempBufAlign;
    }

    if constexpr (CombineQuantMode != COMBINE_NO_QUANT) {
        uint32_t scaleNum = Ops::Base::CeilDiv(static_cast<uint32_t>(k_), static_cast<uint32_t>(ALIGN_32));
        // Tensor用处：DeQuantMxFp8 中用于存储 bf16 格式的 scale（e8m0 转换后的中间结果）
        // Tensor大小：scaleNum × sizeof(bfloat16_t) × DOUBLE_BUFFER × HALF_TO_FP32
        uint32_t bf16ScaleBufAlign =
            Ops::Base::CeilAlign(static_cast<uint32_t>(scaleNum * sizeof(bfloat16_t) * DOUBLE_BUFFER * HALF_TO_FP32),
                                 static_cast<uint32_t>(ALIGN_32));
        bf16ScaleTensor_ =
            LocalTensor<bfloat16_t>(TPosition::VECCALC, tempAddr, bf16ScaleBufAlign / sizeof(bfloat16_t));
        tempAddr += bf16ScaleBufAlign;
        // Tensor用处：DeQuantMxFp8 中用于存储 fp32 格式的 scale（广播后的最终 scale）
        // Tensor大小：scaleNum × sizeof(float) × DOUBLE_BUFFER × HALF_TO_FP32
        uint32_t fp32ScaleBufAlign =
            Ops::Base::CeilAlign(static_cast<uint32_t>(scaleNum * sizeof(float) * DOUBLE_BUFFER * HALF_TO_FP32),
                                 static_cast<uint32_t>(ALIGN_32));
        fp32ScaleTensor_ = LocalTensor<float>(TPosition::VECCALC, tempAddr, fp32ScaleBufAlign / sizeof(float));
        tempAddr += fp32ScaleBufAlign;
    }

    return tokensPerBatch;
}

// ===============================================================
// Unpermute：主入口 — 初始化 buffer → 分批循环处理
// ===============================================================
template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::Unpermute()
{
    int32_t coreLen, coreOffset;
    TilingByCore(m_, coreLen, coreOffset, 1);
    if (coreLen == 0) {
        return;
    }
    int32_t tokensPerBatch = UnpermuteBuffInit(coreLen);

    GlobalTensor<bfloat16_t> expandedX;
    expandedX.SetGlobalBuffer((__gm__ bfloat16_t *)params_.peermemInfo.combineSendPtr);
    GlobalTensor<bfloat16_t> output;
    output.SetGlobalBuffer((__gm__ bfloat16_t *)params_.y2GmAddr);

    // 外层：token batch
    for (int32_t batchTokenOffset = 0; batchTokenOffset < coreLen; batchTokenOffset += tokensPerBatch) {
        int32_t batchTokenCount =
            (batchTokenOffset + tokensPerBatch > coreLen) ? (coreLen - batchTokenOffset) : tokensPerBatch;

        UnpermuteLoadWeights(coreOffset, batchTokenOffset, batchTokenCount, topKWeightsBf16Tensor_);

        // 内层：token 循环
        SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
        for (int32_t localIdx = 0; localIdx < batchTokenCount; localIdx++) {
            int32_t tokenIdx = coreOffset + batchTokenOffset + localIdx;
            SyncFuncStatic<AscendC::HardEvent::MTE3_MTE2, SYNC_EVENT_ID2>();
            UnpermuteProcessToken(tokenIdx, localIdx, expandedX);
            Cast(dataResTensor_, dataResFp32Tensor_, AscendC::RoundMode::CAST_RINT, k_);
            SyncFuncStatic<AscendC::HardEvent::V_MTE3, SYNC_EVENT_ID3>();
            DataCopy(output[static_cast<uint64_t>(tokenIdx) * k_], dataResTensor_, k_);
        }
        WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
    }
}

#endif // MEGA_MOE_COMBINE_REDUCE_H
