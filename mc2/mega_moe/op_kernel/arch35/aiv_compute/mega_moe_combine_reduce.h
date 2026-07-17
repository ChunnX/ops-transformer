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
#include "mega_moe_combine_send_compute.h"

using namespace AscendC;

namespace MegaMoeCombineReduceImpl {
// 连续逻辑 tile 的调度信息；tileId 不与物理核 ID 绑定。
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
// ComputeCombineReduceTileRange：将 m 个 token 均匀分配给连续逻辑 tile
// ===============================================================
__aicore__ inline __attribute__((always_inline)) void ComputeCombineReduceTileRange(
    const CombineReduceTileSchedule& tileSchedule,
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

// ===============================================================
// InitCombineReduceWorkspace：分配 CombineReduce 所需固定 buffer，返回 batchLength
// ===============================================================
template <int32_t CombineQuantMode, typename TopkWeightsType>
__aicore__ inline __attribute__((always_inline)) int32_t InitCombineReduceWorkspace(int32_t tileLength,
    const CombineReduceShapeContext& shapeContext, CombineReduceWorkspace& workspace)
{
    // dataResTensor：3 个 token 长度，2 块用于 MTE2 double buffer，1 块存储 Cast 后的搬出结果。
    uint32_t dataResBufAlign = Ops::Base::CeilAlign(
        static_cast<uint32_t>(UNPERMUTE_LIST_NUM * shapeContext.k * sizeof(bfloat16_t)),
        static_cast<uint32_t>(ALIGN_32));
    uint32_t dataResFp32BufAlign = dataResBufAlign * HALF_TO_FP32;
    uint32_t dataResAddr = 0;
    workspace.dataResTensor = LocalTensor<bfloat16_t>(TPosition::VECCALC, dataResAddr,
        dataResBufAlign / sizeof(bfloat16_t));
    // dataResFp32Tensor：保存 token Cast 结果及 topK 加权累加结果。
    uint32_t dataResFp32Addr = dataResAddr + dataResBufAlign;
    workspace.dataResFp32Tensor = LocalTensor<float>(TPosition::VECCALC, dataResFp32Addr,
        dataResFp32BufAlign / sizeof(float));
    uint32_t tempAddr = dataResFp32Addr + dataResFp32BufAlign;

    // 权重 buffer 按 batch 分配；topK 超过基准值时缩小 batch，
    // 保持 UB 占用不超过基准预算。
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
        // bf16 weight 的 GM→UB 搬运中转 buffer，后续 Cast 为 fp32。
        uint32_t tempBufAlign = Ops::Base::CeilAlign(
            static_cast<uint32_t>(batchLength * shapeContext.topK * sizeof(bfloat16_t)),
            static_cast<uint32_t>(ALIGN_32));
        workspace.topKWeightsBf16Tensor = LocalTensor<bfloat16_t>(TPosition::VECCALC, tempAddr,
            tempBufAlign / sizeof(bfloat16_t));
        tempAddr += tempBufAlign;
    }

    if constexpr (CombineQuantMode != COMBINE_NO_QUANT) {
        // DeQuantMxFp8 使用的 bf16 scale 中间结果和 fp32 scale 广播结果。
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

// ===============================================================
// LoadCombineReduceWeights：加载一个 token batch 的 topK 权重到 UB
// ===============================================================
template <typename TopkWeightsType>
__aicore__ inline __attribute__((always_inline)) void LoadCombineReduceWeights(
    int32_t tileOffset, int32_t batchStart, int32_t tokenCount,
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

// ===============================================================
// ProcessCombineReduceToken：对单个 token 的 topK expert 结果做反量化、加权和累加
// ===============================================================
template <int32_t CombineQuantMode>
__aicore__ inline __attribute__((always_inline)) void ProcessCombineReduceToken(
    int32_t tokenIdx, int32_t localIdx, uint32_t k, uint32_t topK,
    const GlobalTensor<bfloat16_t>& expandedX,
    CombineReduceWorkspace& workspace)
{
    LocalTensor<bfloat16_t> dataResTensor = workspace.dataResTensor;
    LocalTensor<float> dataResFp32Tensor = workspace.dataResFp32Tensor;
    LocalTensor<float> topKWeightsTensor = workspace.topKWeightsTensor;
    LocalTensor<bfloat16_t> dataIn0Bf16 = dataResTensor[k];
    LocalTensor<bfloat16_t> dataIn1Bf16 = dataResTensor[k * 2];
    LocalTensor<float> dataIn0Fp32 = dataResFp32Tensor[k];
    LocalTensor<float> dataIn1Fp32 = dataResFp32Tensor[k * 2];
    for (int32_t expertId = 0; expertId < static_cast<int32_t>(topK); ++expertId) {
        float expertScale = topKWeightsTensor.GetValue(localIdx * topK + expertId);
        auto event = (expertId % DOUBLE_BUFFER == 0) ? EVENT_ID0 : EVENT_ID1;
        auto dataInBf16 = (expertId % DOUBLE_BUFFER == 0) ? dataIn0Bf16 : dataIn1Bf16;
        auto dataInFp32 = (expertId % DOUBLE_BUFFER == 0) ? dataIn0Fp32 : dataIn1Fp32;
        if constexpr (CombineQuantMode == COMBINE_NO_QUANT) {
            // 非量化路径：直接搬入 bf16 token 并 Cast 为 fp32。
            WaitFlag<AscendC::HardEvent::V_MTE2>(event);
            DataCopy(dataInBf16,
                expandedX[(static_cast<uint64_t>(tokenIdx) * topK + expertId) * k], k);
            SetFlag<AscendC::HardEvent::MTE2_V>(event);
            WaitFlag<AscendC::HardEvent::MTE2_V>(event);
            SetFlag<AscendC::HardEvent::S_V>(event);
            WaitFlag<AscendC::HardEvent::S_V>(event);
            Cast(dataInFp32, dataInBf16, AscendC::RoundMode::CAST_NONE, k);
        } else {
            // 量化路径：搬入 FP8 token+scale，反量化为 fp32。
            uint32_t scaleNum = Ops::Base::CeilDiv(k, uint32_t(MXFP_SCALE_GROUP_NUM));
            uint32_t quantTokenSize = k + scaleNum;
            uint32_t quantElementNum = quantTokenSize / sizeof(bfloat16_t);
            WaitFlag<AscendC::HardEvent::V_MTE2>(event);
            DataCopy(dataInBf16,
                expandedX[(static_cast<uint64_t>(tokenIdx) * topK + expertId) * quantElementNum],
                quantElementNum);
            SetFlag<AscendC::HardEvent::MTE2_V>(event);
            WaitFlag<AscendC::HardEvent::MTE2_V>(event);
            using Fp8Type = typename std::conditional<CombineQuantMode == MXFP8_E4M3_COMM_QUANT,
                fp8_e4m3fn_t, fp8_e5m2_t>::type;
            LocalTensor<bfloat16_t> bf16ScaleTensor = workspace.bf16ScaleTensor;
            LocalTensor<float> fp32ScaleTensor = workspace.fp32ScaleTensor;
            MegaMoeCombineImpl::DeQuantMxFp8<Fp8Type, bfloat16_t>(dataInBf16, dataInFp32,
                bf16ScaleTensor, fp32ScaleTensor, scaleNum, k);
        }
        PipeBarrier<PIPE_V>();
        // 将每个 expert 的输出乘以对应 topK weight，累加到第 0 块 fp32 buffer。
        if (expertId == 0) {
            Muls(dataResFp32Tensor, dataInFp32, expertScale, k);
        } else {
            Muls(dataInFp32, dataInFp32, expertScale, k);
            PipeBarrier<PIPE_V>();
            Add(dataResFp32Tensor, dataResFp32Tensor, dataInFp32, k);
            PipeBarrier<PIPE_V>();
        }
        SetFlag<AscendC::HardEvent::V_MTE2>(event);
    }
}

// ===============================================================
// ProcessCombineReduceSingleCore：单核入口——分核、初始化 buffer，再按 batch 执行 CombineReduce
// ===============================================================
template <int32_t CombineQuantMode, typename TopkWeightsType>
__aicore__ inline __attribute__((always_inline)) void ProcessCombineReduceSingleCore(
    const CombineReduceTileSchedule& tileSchedule,
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

    // 外层：token batch，权重 buffer 在 batch 之间复用。
    for (int32_t batchStart = 0; batchStart < tileLength; batchStart += batchLength) {
        int32_t tokenCount = (batchStart + batchLength > tileLength) ?
            (tileLength - batchStart) : batchLength;
        LoadCombineReduceWeights<TopkWeightsType>(
            tileOffset, batchStart, tokenCount, shapeContext, bufferContext, workspace);
        // 内层：token 循环，EVENT_ID0/1 用于 expert 数据搬运的 double buffer。
        SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
        for (int32_t localIdx = 0; localIdx < tokenCount; ++localIdx) {
            int32_t tokenIdx = tileOffset + batchStart + localIdx;
            SyncFuncStatic<AscendC::HardEvent::MTE3_MTE2, SYNC_EVENT_ID2>();
            ProcessCombineReduceToken<CombineQuantMode>(
                tokenIdx, localIdx, shapeContext.k, shapeContext.topK, expandedX, workspace);
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

} // namespace MegaMoeCombineReduceImpl

#endif // MEGA_MOE_COMBINE_REDUCE_H
