/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MEGA_MOE_UNPERMUTE_H
#define MEGA_MOE_UNPERMUTE_H

#include "../common/mega_moe_mxfp8_utils.h"
#include "../common/mega_moe_utils.h"

namespace MegaMoeImpl {

using namespace AscendC;

struct TokenUnpermuteConfig {
    AivJobContext job;
    uint32_t quantTokenSizeBytes;
    uint32_t fullTokenChunkJobCount;
    MegaMoeUnpermuteBufferConfig fullTokenChunkConfig;
    MegaMoeUnpermuteBufferConfig tailTokenChunkConfig;
};

struct TokenUnpermuteScratch {
    LocalTensor<bfloat16_t> dataResTensor;
    LocalTensor<float> dataResFp32Tensor;
    LocalTensor<float> topKWeightsTensor;
    LocalTensor<float> fp32ScaleTensor;
    LocalTensor<bfloat16_t> bf16ScaleTensor;
    LocalTensor<bfloat16_t> topKWeightsBf16Tensor;
};

// metaInfo 在 Dispatch 写入后由 GMM2/Combine 只读，并持续有效到算子结束。
// 在所有 rank 的 Combine 发送完成后，将稳定的紧凑行 metadata 复制到调试输出尾段：
// [maxOutputSize, hidden] BF16 + [maxOutputSize, META_INFO_SIZE] INT32。
__aicore__ inline void ExportGmm2DebugMetaInfo(uint32_t tokenHiddenDim, const Params &params,
                                               const AivJobContext &job)
{
    if constexpr (g_coreType == AIC) {
        return;
    }
    if (!params.tilingData->activationDebug || params.activationDebugOutGmAddr == nullptr || job.totalJobs == 0U) {
        return;
    }
    WorkRange rowRange = TilingByJobContext(params.tilingData->maxOutputSize, job.jobIndex, job.totalJobs, 1U);
    if (rowRange.count == 0U) {
        return;
    }

    GlobalTensor<int32_t> metaInfoSource;
    metaInfoSource.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(params.workspaceInfo.metaInfoPtr));
    uint64_t dataBytes = static_cast<uint64_t>(params.tilingData->maxOutputSize) * tokenHiddenDim *
                         sizeof(bfloat16_t);
    GlobalTensor<int32_t> metaInfoDebug;
    metaInfoDebug.SetGlobalBuffer(
        reinterpret_cast<__gm__ int32_t *>(params.activationDebugOutGmAddr + dataBytes));

    constexpr uint32_t ROWS_PER_COPY = 64U;
    LocalTensor<int32_t> metaInfoUb(TPosition::VECCALC, 0U, ROWS_PER_COPY * META_INFO_SIZE);
    uint64_t rowEnd = rowRange.start + rowRange.count;
    for (uint64_t rowStart = rowRange.start; rowStart < rowEnd; rowStart += ROWS_PER_COPY) {
        uint32_t rowCount = static_cast<uint32_t>(rowEnd - rowStart);
        rowCount = rowCount < ROWS_PER_COPY ? rowCount : ROWS_PER_COPY;
        uint32_t copyElements = rowCount * META_INFO_SIZE;
        uint64_t elementOffset = rowStart * META_INFO_SIZE;
        DataCopy(metaInfoUb, metaInfoSource[elementOffset], copyElements);
        SyncFuncStatic<HardEvent::MTE2_MTE3, SYNC_EVENT_ID1>();
        DataCopy(metaInfoDebug[elementOffset], metaInfoUb, copyElements);
        SyncFuncStatic<HardEvent::MTE3_MTE2, SYNC_EVENT_ID1>();
    }
}

// 将 Combine 回源并完成通信反量化、但尚未乘 top-k 权重和求和的 FP32 槽位写入调试输出尾段。
// 调试输出布局：
// [maxOutputSize, hidden] BF16 + [maxOutputSize, META_INFO_SIZE] INT32 +
// [tokenNum, topK, hidden] FP32 + [tokenNum, hidden] FP32 +
// [tokenNum, sharedExpertNum, hidden] FP32 + [tokenNum, hidden] FP32。
__aicore__ inline void ExportCombineDebugSlot(uint32_t tokenHiddenDim, uint32_t topK, const Params &params,
                                              int32_t tokenIdx, int32_t topkIdx,
                                              const LocalTensor<float> &slotFp32)
{
    if (!params.tilingData->activationDebug || params.activationDebugOutGmAddr == nullptr) {
        return;
    }
    uint64_t gmm2DataBytes = static_cast<uint64_t>(params.tilingData->maxOutputSize) * tokenHiddenDim *
                             sizeof(bfloat16_t);
    uint64_t metaBytes = static_cast<uint64_t>(params.tilingData->maxOutputSize) * META_INFO_SIZE * sizeof(int32_t);
    GlobalTensor<float> combineDebug;
    combineDebug.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(
        params.activationDebugOutGmAddr + gmm2DataBytes + metaBytes));
    uint64_t slotOffset =
        (static_cast<uint64_t>(tokenIdx) * topK + static_cast<uint64_t>(topkIdx)) * tokenHiddenDim;
    SyncFuncStatic<HardEvent::V_MTE3, SYNC_EVENT_ID4>();
    DataCopy(combineDebug[slotOffset], slotFp32, tokenHiddenDim);
    SyncFuncStatic<HardEvent::MTE3_V, SYNC_EVENT_ID4>();
}

// 将 top-k 路由专家完成权重乘法和 FP32 累加、但尚未加入共享专家的结果导出。
__aicore__ inline void ExportMoeSumDebug(uint32_t tokenHiddenDim, uint32_t topK, uint32_t tokenNum,
                                         const Params &params, int32_t tokenIdx,
                                         const LocalTensor<float> &moeSumFp32)
{
    if (!params.tilingData->activationDebug || params.activationDebugOutGmAddr == nullptr) {
        return;
    }
    uint64_t gmm2DataBytes = static_cast<uint64_t>(params.tilingData->maxOutputSize) * tokenHiddenDim *
                             sizeof(bfloat16_t);
    uint64_t metaBytes = static_cast<uint64_t>(params.tilingData->maxOutputSize) * META_INFO_SIZE * sizeof(int32_t);
    uint64_t combineBytes = static_cast<uint64_t>(tokenNum) * topK * tokenHiddenDim * sizeof(float);
    GlobalTensor<float> moeSumDebug;
    moeSumDebug.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(
        params.activationDebugOutGmAddr + gmm2DataBytes + metaBytes + combineBytes));
    uint64_t tokenOffset = static_cast<uint64_t>(tokenIdx) * tokenHiddenDim;
    SyncFuncStatic<HardEvent::V_MTE3, SYNC_EVENT_ID4>();
    DataCopy(moeSumDebug[tokenOffset], moeSumFp32, tokenHiddenDim);
    SyncFuncStatic<HardEvent::MTE3_V, SYNC_EVENT_ID4>();
}

// 将共享专家 GMM2 结果从 BF16 转成 FP32 后、加入 MoE 累加器前逐专家导出。
__aicore__ inline void ExportSharedExpertDebug(uint32_t tokenHiddenDim, uint32_t topK, uint32_t tokenNum,
                                               uint32_t sharedExpertNum, const Params &params, int32_t tokenIdx,
                                               uint32_t sharedExpertIdx,
                                               const LocalTensor<float> &sharedFp32)
{
    if (!params.tilingData->activationDebug || params.activationDebugOutGmAddr == nullptr || sharedExpertNum == 0U) {
        return;
    }
    uint64_t gmm2DataBytes = static_cast<uint64_t>(params.tilingData->maxOutputSize) * tokenHiddenDim *
                             sizeof(bfloat16_t);
    uint64_t metaBytes = static_cast<uint64_t>(params.tilingData->maxOutputSize) * META_INFO_SIZE * sizeof(int32_t);
    uint64_t combineBytes = static_cast<uint64_t>(tokenNum) * topK * tokenHiddenDim * sizeof(float);
    uint64_t moeSumBytes = static_cast<uint64_t>(tokenNum) * tokenHiddenDim * sizeof(float);
    GlobalTensor<float> sharedDebug;
    sharedDebug.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(
        params.activationDebugOutGmAddr + gmm2DataBytes + metaBytes + combineBytes + moeSumBytes));
    uint64_t slotOffset =
        (static_cast<uint64_t>(tokenIdx) * sharedExpertNum + sharedExpertIdx) * tokenHiddenDim;
    SyncFuncStatic<HardEvent::V_MTE3, SYNC_EVENT_ID4>();
    DataCopy(sharedDebug[slotOffset], sharedFp32, tokenHiddenDim);
    SyncFuncStatic<HardEvent::MTE3_V, SYNC_EVENT_ID4>();
}

// 将共享专家全部加入完成后的 FP32 总和导出；该值紧邻最终 FP32 -> BF16 cast。
__aicore__ inline void ExportPreCastTotalDebug(uint32_t tokenHiddenDim, uint32_t topK, uint32_t tokenNum,
                                               uint32_t sharedExpertNum, const Params &params, int32_t tokenIdx,
                                               const LocalTensor<float> &preCastTotalFp32)
{
    if (!params.tilingData->activationDebug || params.activationDebugOutGmAddr == nullptr) {
        return;
    }
    uint64_t gmm2DataBytes = static_cast<uint64_t>(params.tilingData->maxOutputSize) * tokenHiddenDim *
                             sizeof(bfloat16_t);
    uint64_t metaBytes = static_cast<uint64_t>(params.tilingData->maxOutputSize) * META_INFO_SIZE * sizeof(int32_t);
    uint64_t combineBytes = static_cast<uint64_t>(tokenNum) * topK * tokenHiddenDim * sizeof(float);
    uint64_t moeSumBytes = static_cast<uint64_t>(tokenNum) * tokenHiddenDim * sizeof(float);
    uint64_t sharedBytes = static_cast<uint64_t>(tokenNum) * sharedExpertNum * tokenHiddenDim * sizeof(float);
    GlobalTensor<float> preCastTotalDebug;
    preCastTotalDebug.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(
        params.activationDebugOutGmAddr + gmm2DataBytes + metaBytes + combineBytes + moeSumBytes + sharedBytes));
    uint64_t tokenOffset = static_cast<uint64_t>(tokenIdx) * tokenHiddenDim;
    SyncFuncStatic<HardEvent::V_MTE3, SYNC_EVENT_ID4>();
    DataCopy(preCastTotalDebug[tokenOffset], preCastTotalFp32, tokenHiddenDim);
    SyncFuncStatic<HardEvent::MTE3_V, SYNC_EVENT_ID4>();
}

// 加载并转换一批 top-k 权重。
template <typename TopkWeightsType>
__aicore__ inline void LoadTokenUnpermuteWeights(const MoeStageCommonConfig &common, const Params &params,
                                                 TokenUnpermuteScratch &scratch, int32_t jobOffset,
                                                 int32_t batchTokenOffset, int32_t batchTokenCount)
{
    if constexpr (Std::IsSame<TopkWeightsType, float>::value) {
        GlobalTensor<float> topKWeightsGm;
        topKWeightsGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(params.probsGmAddr));
        DataCopyPad(scratch.topKWeightsTensor,
                    topKWeightsGm[static_cast<uint64_t>(jobOffset + batchTokenOffset) * common.topK],
                    {1U, static_cast<uint32_t>(batchTokenCount * common.topK * sizeof(float)), 0U, 0U, 0U},
                    {false, 0U, 0U, 0U});
        SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
    }
    if constexpr (Std::IsSame<TopkWeightsType, bfloat16_t>::value) {
        GlobalTensor<bfloat16_t> topKWeightsGm;
        topKWeightsGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(params.probsGmAddr));
        DataCopyPad(scratch.topKWeightsBf16Tensor,
                    topKWeightsGm[static_cast<uint64_t>(jobOffset + batchTokenOffset) * common.topK],
                    {1U, static_cast<uint32_t>(batchTokenCount * common.topK * sizeof(bfloat16_t)), 0U, 0U, 0U},
                    {false, 0U, 0U, 0U});
        SyncFuncStatic<HardEvent::MTE2_V, SYNC_EVENT_ID2>();
        Cast(scratch.topKWeightsTensor, scratch.topKWeightsBf16Tensor, RoundMode::CAST_NONE,
             batchTokenCount * common.topK);
        SetFlag<HardEvent::V_S>(EVENT_ID0);
        WaitFlag<HardEvent::V_S>(EVENT_ID0);
    }
}

// 将一个 MoE 专家的输入搬入 UB，并按 Combine 模式完成 Cast 或 MXFP8 反量化。
template <uint8_t CombineMode>
__aicore__ inline void LoadMoeExpertInput(const TokenUnpermuteConfig &context, const MoeStageCommonConfig &common,
                                          TokenUnpermuteScratch &scratch, const GlobalTensor<bfloat16_t> &expandedX,
                                          uint64_t expertInputIndex, TEventID event,
                                          LocalTensor<bfloat16_t> &dataInBf16, LocalTensor<float> &dataInFp32)
{
    WaitFlag<HardEvent::V_MTE2>(event);
    if constexpr (CombineMode == COMBINE_NO_QUANT) {
        DataCopy(dataInBf16, expandedX[expertInputIndex * common.tokenHiddenDim], common.tokenHiddenDim);
        SetFlag<HardEvent::MTE2_V>(event);
        WaitFlag<HardEvent::MTE2_V>(event);
        Cast(dataInFp32, dataInBf16, RoundMode::CAST_NONE, common.tokenHiddenDim);
    } else {
        uint32_t quantTokenElementCount = context.quantTokenSizeBytes / sizeof(bfloat16_t);
        DataCopy(dataInBf16, expandedX[expertInputIndex * quantTokenElementCount], quantTokenElementCount);
        SetFlag<HardEvent::MTE2_V>(event);
        WaitFlag<HardEvent::MTE2_V>(event);
        uint32_t nScale = Ops::Base::CeilDiv(common.tokenHiddenDim, static_cast<uint32_t>(MXFP_SCALE_GROUP_NUM));
        using Fp8Type = typename std::conditional<CombineMode == MXFP8_E4M3_COMM_QUANT, fp8_e4m3fn_t, fp8_e5m2_t>::type;
        Mxfp8::DeQuantMxFp8<Fp8Type, bfloat16_t>(dataInBf16, dataInFp32, scratch.bf16ScaleTensor,
                                                 scratch.fp32ScaleTensor, nScale, common.tokenHiddenDim);
    }
}

// 累加一个 token 对应的全部 MoE 专家输入。
template <uint8_t CombineMode, bool TopkWeightsPrefetch>
__aicore__ inline void AccumulateMoeExpertsForToken(const TokenUnpermuteConfig &context,
                                                    const MoeStageCommonConfig &common, const Params &params,
                                                    TokenUnpermuteScratch &scratch,
                                                    int32_t tokenIdx, int32_t localIdx,
                                                    const GlobalTensor<bfloat16_t> &expandedX,
                                                    const MegaMoeUnpermuteBufferConfig &bufferConfig)
{
    for (int32_t expertIdx = 0; expertIdx < static_cast<int32_t>(common.topK); ++expertIdx) {
        int32_t accumulationItemIdx = localIdx * static_cast<int32_t>(common.topK + common.sharedExpertNum) + expertIdx;
        int32_t inputBufferIdx = accumulationItemIdx % bufferConfig.inputBufferCount;
        TEventID event = static_cast<TEventID>(inputBufferIdx);
        LocalTensor<bfloat16_t> dataInBf16 =
            scratch.dataResTensor[(inputBufferIdx + 1) * bufferConfig.bf16SlotElementCount];
        LocalTensor<float> dataInFp32 =
            scratch.dataResFp32Tensor[(inputBufferIdx + 1) * bufferConfig.fp32SlotElementCount];
        uint64_t expertInputIndex = static_cast<uint64_t>(tokenIdx) * common.topK + expertIdx;
        LoadMoeExpertInput<CombineMode>(context, common, scratch, expandedX, expertInputIndex, event, dataInBf16,
                                        dataInFp32);
        SetFlag<HardEvent::S_V>(event);
        WaitFlag<HardEvent::S_V>(event);
        PipeBarrier<PIPE_V>();
        ExportCombineDebugSlot(common.tokenHiddenDim, common.topK, params, tokenIdx, expertIdx, dataInFp32);
        if (expertIdx == 0) {
            if constexpr (TopkWeightsPrefetch) {
                DataCopy(scratch.dataResFp32Tensor, dataInFp32, common.tokenHiddenDim);
            } else {
                float expertScale = scratch.topKWeightsTensor.GetValue(localIdx * common.topK + expertIdx);
                Muls(scratch.dataResFp32Tensor, dataInFp32, expertScale, common.tokenHiddenDim);
            }
        } else {
            if constexpr (!TopkWeightsPrefetch) {
                float expertScale = scratch.topKWeightsTensor.GetValue(localIdx * common.topK + expertIdx);
                Muls(dataInFp32, dataInFp32, expertScale, common.tokenHiddenDim);
                PipeBarrier<PIPE_V>();
            }
            Add(scratch.dataResFp32Tensor, scratch.dataResFp32Tensor, dataInFp32, common.tokenHiddenDim);
            PipeBarrier<PIPE_V>();
        }
        SetFlag<HardEvent::V_MTE2>(event);
    }
}

// 等待该共享专家的 GMM2 结果就绪后累加进当前 token（等待与累加是一体动作，调用方无需分两步）。
template <uint32_t Gmm1TileM>
__aicore__ inline void AccumulateSharedExpertForToken(const MoeStageCommonConfig &common, const Params &params,
                                                      TokenUnpermuteScratch &scratch, int32_t tokenIdx,
                                                      int32_t localIdx, uint32_t sharedExpertIdx,
                                                      const MegaMoeUnpermuteBufferConfig &bufferConfig)
{
    uint32_t tokenGroupIndex = static_cast<uint32_t>(tokenIdx) / Gmm1TileM;
    uint32_t tokenGroupCount = Ops::Base::CeilDiv(common.tokenNum, Gmm1TileM);
    uint64_t sharedExpertStride = static_cast<uint64_t>(tokenGroupCount) * INT_CACHELINE;
    __gm__ int32_t *counterAddr = GetCombineSyncCounterAddress(
        reinterpret_cast<__gm__ int32_t *>(params.workspaceInfo.sharedExpertGmm2TileCounterPtr) +
            static_cast<uint64_t>(sharedExpertIdx) * sharedExpertStride,
        tokenGroupIndex);
    uint32_t gmm2NTilesPerGroup = Ops::Base::CeilDiv(common.tokenHiddenDim, L1_TILE_N);
    WaitUntilGmFlagEquals(counterAddr, static_cast<int32_t>(gmm2NTilesPerGroup));

    GlobalTensor<bfloat16_t> sharedResult;
    sharedResult.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(params.workspaceInfo.sharedExpertResultPtr));
    int32_t accumulationItemIdx = localIdx * static_cast<int32_t>(common.topK + common.sharedExpertNum) +
                                  static_cast<int32_t>(common.topK + sharedExpertIdx);
    int32_t inputBufferIdx = accumulationItemIdx % bufferConfig.inputBufferCount;
    TEventID event = static_cast<TEventID>(inputBufferIdx);
    LocalTensor<bfloat16_t> dataInBf16 =
        scratch.dataResTensor[(inputBufferIdx + 1) * bufferConfig.bf16SlotElementCount];
    LocalTensor<float> dataInFp32 = scratch.dataResFp32Tensor[(inputBufferIdx + 1) * bufferConfig.fp32SlotElementCount];
    WaitFlag<HardEvent::V_MTE2>(event);
    DataCopy(
        dataInBf16,
        sharedResult[(static_cast<uint64_t>(sharedExpertIdx) * common.tokenNum + tokenIdx) * common.tokenHiddenDim],
        common.tokenHiddenDim);
    SetFlag<HardEvent::MTE2_V>(event);
    WaitFlag<HardEvent::MTE2_V>(event);
    SetFlag<HardEvent::S_V>(event);
    WaitFlag<HardEvent::S_V>(event);
    Cast(dataInFp32, dataInBf16, RoundMode::CAST_NONE, common.tokenHiddenDim);
    PipeBarrier<PIPE_V>();
    ExportSharedExpertDebug(common.tokenHiddenDim, common.topK, common.tokenNum, common.sharedExpertNum,
                            params, tokenIdx, sharedExpertIdx, dataInFp32);
    Add(scratch.dataResFp32Tensor, scratch.dataResFp32Tensor, dataInFp32, common.tokenHiddenDim);
    PipeBarrier<PIPE_V>();
    SetFlag<HardEvent::V_MTE2>(event);
}

template <uint8_t CombineMode, typename TopkWeightsType, bool TopkWeightsPrefetch, uint32_t Gmm1TileM>
__aicore__ inline void ProcessTokenUnpermuteBatch(
    const TokenUnpermuteConfig &context, const MoeStageCommonConfig &common, const Params &params,
    TokenUnpermuteScratch &scratch, const MegaMoeUnpermuteBufferConfig &bufferConfig,
    const GlobalTensor<bfloat16_t> &expandedX, GlobalTensor<bfloat16_t> &output, int32_t jobOffset,
    int32_t batchTokenOffset, int32_t batchTokenCount, TEventID outputBufferEvent)
{
    if constexpr (!TopkWeightsPrefetch) {
        LoadTokenUnpermuteWeights<TopkWeightsType>(common, params, scratch, jobOffset, batchTokenOffset,
                                                   batchTokenCount);
    }
    for (int32_t bufferIdx = 0; bufferIdx < bufferConfig.inputBufferCount; ++bufferIdx) {
        SetFlag<HardEvent::V_MTE2>(static_cast<TEventID>(bufferIdx));
    }
    for (int32_t localIdx = 0; localIdx < batchTokenCount; ++localIdx) {
        int32_t tokenIdx = jobOffset + batchTokenOffset + localIdx;
        AccumulateMoeExpertsForToken<CombineMode, TopkWeightsPrefetch>(context, common, params, scratch, tokenIdx,
                                                                       localIdx, expandedX, bufferConfig);
        ExportMoeSumDebug(common.tokenHiddenDim, common.topK, common.tokenNum, params, tokenIdx,
                          scratch.dataResFp32Tensor);
        for (uint32_t sharedExpertIdx = 0; sharedExpertIdx < common.sharedExpertNum; ++sharedExpertIdx) {
            AccumulateSharedExpertForToken<Gmm1TileM>(common, params, scratch, tokenIdx, localIdx, sharedExpertIdx,
                                                      bufferConfig);
        }
        ExportPreCastTotalDebug(common.tokenHiddenDim, common.topK, common.tokenNum, common.sharedExpertNum,
                                params, tokenIdx, scratch.dataResFp32Tensor);
        WaitFlag<HardEvent::MTE3_V>(outputBufferEvent);
        Cast(scratch.dataResTensor, scratch.dataResFp32Tensor, RoundMode::CAST_RINT, common.tokenHiddenDim);
        SyncFuncStatic<HardEvent::V_MTE3, SYNC_EVENT_ID3>();
        DataCopy(output[static_cast<uint64_t>(tokenIdx) * common.tokenHiddenDim], scratch.dataResTensor,
                 common.tokenHiddenDim);
        SetFlag<HardEvent::MTE3_V>(outputBufferEvent);
    }
    for (int32_t bufferIdx = 0; bufferIdx < bufferConfig.inputBufferCount; ++bufferIdx) {
        WaitFlag<HardEvent::V_MTE2>(static_cast<TEventID>(bufferIdx));
    }
}

/*
 * 构造 Unpermute 使用的 UB 视图，并返回当前 AIV 对应的 buffer 配置（普通/wave/a8w4 编排共用）。
 * UB 从地址 0 起按下列顺序连续排布（N = inputBufferCount，每段起点=上一段末尾）：
 *   [dataRes    ] (N+1) 个 bf16 槽 × bf16SlotElementCount —— 累加结果 bf16 输出
 *   [dataResFp32] (N+1) 个 fp32 槽 × fp32SlotElementCount —— fp32 累加中间态
 *   [topKWeights] topKWeightsBufferBytes —— 本批 token 的 topk 权重（fp32）
 *   [权重转换区 ] 仅 TopkWeightsType==bf16 时存在（topKWeightsConversionBufferBytes）
 *   [bf16Scale  ] 仅量化 combine 时存在，Align32(ceil(H/32)·2B·扩展系数)
 *   [fp32Scale  ] 仅量化 combine 时存在，Align32(ceil(H/32)·4B·扩展系数)
 */
template <typename TopkWeightsType, int32_t CombineQuantMode>
__aicore__ inline MegaMoeUnpermuteBufferConfig CreateTokenUnpermuteBuffers(const TokenUnpermuteConfig &context,
                                                                           uint32_t tokenHiddenDim,
                                                                           TokenUnpermuteScratch &scratch)
{
    MegaMoeUnpermuteBufferConfig bufferConfig = context.job.jobIndex < context.fullTokenChunkJobCount ?
                                                    context.fullTokenChunkConfig :
                                                    context.tailTokenChunkConfig;

    uint32_t bf16ScaleBufAlign = 0U;
    uint32_t fp32ScaleBufAlign = 0U;
    if constexpr (CombineQuantMode != COMBINE_NO_QUANT) {
        uint32_t scaleNum = Ops::Base::CeilDiv(tokenHiddenDim, static_cast<uint32_t>(ALIGN_32));
        bf16ScaleBufAlign = Ops::Base::CeilAlign(
            scaleNum * static_cast<uint32_t>(sizeof(bfloat16_t)) * static_cast<uint32_t>(DEQUANT_BF16_SCALE_EXPANSION),
            static_cast<uint32_t>(ALIGN_32));
        fp32ScaleBufAlign = Ops::Base::CeilAlign(
            scaleNum * static_cast<uint32_t>(sizeof(float)) * static_cast<uint32_t>(DEQUANT_FP32_SCALE_EXPANSION),
            static_cast<uint32_t>(ALIGN_32));
    }

    uint32_t bf16SlotBytes = bufferConfig.bf16SlotElementCount * sizeof(bfloat16_t);
    uint32_t fp32SlotBytes = bufferConfig.fp32SlotElementCount * sizeof(float);
    uint32_t dataResBufBytes = (bufferConfig.inputBufferCount + 1) * bf16SlotBytes;
    uint32_t dataResFp32BufBytes = (bufferConfig.inputBufferCount + 1) * fp32SlotBytes;
    uint32_t dataResAddr = 0U;
    scratch.dataResTensor =
        LocalTensor<bfloat16_t>(TPosition::VECCALC, dataResAddr, dataResBufBytes / sizeof(bfloat16_t));
    uint32_t dataResFp32Addr = dataResAddr + dataResBufBytes;
    scratch.dataResFp32Tensor =
        LocalTensor<float>(TPosition::VECCALC, dataResFp32Addr, dataResFp32BufBytes / sizeof(float));
    uint32_t tempAddr = dataResFp32Addr + dataResFp32BufBytes;

    scratch.topKWeightsTensor =
        LocalTensor<float>(TPosition::VECCALC, tempAddr, bufferConfig.topKWeightsBufferBytes / sizeof(float));
    tempAddr += bufferConfig.topKWeightsBufferBytes;
    if constexpr (Std::IsSame<TopkWeightsType, bfloat16_t>::value) {
        scratch.topKWeightsBf16Tensor = LocalTensor<bfloat16_t>(
            TPosition::VECCALC, tempAddr, bufferConfig.topKWeightsConversionBufferBytes / sizeof(bfloat16_t));
        tempAddr += bufferConfig.topKWeightsConversionBufferBytes;
    }
    if constexpr (CombineQuantMode != COMBINE_NO_QUANT) {
        scratch.bf16ScaleTensor =
            LocalTensor<bfloat16_t>(TPosition::VECCALC, tempAddr, bf16ScaleBufAlign / sizeof(bfloat16_t));
        tempAddr += bf16ScaleBufAlign;
        scratch.fp32ScaleTensor = LocalTensor<float>(TPosition::VECCALC, tempAddr, fp32ScaleBufAlign / sizeof(float));
    }
    return bufferConfig;
}

// 原型：MegaMoe::Unpermute。执行当前 token 分区的完整 Unpermute 流水。
template <uint8_t CombineMode, typename TopkWeightsType, bool TopkWeightsPrefetch, uint32_t Gmm1TileM>
__aicore__ inline void UnpermuteTokens(const TokenUnpermuteConfig &context, const MoeStageCommonConfig &common,
                                       const Params &params, TokenUnpermuteScratch &scratch,
                                       const MegaMoeUnpermuteBufferConfig &bufferConfig)
{
    if constexpr (g_coreType == AIC) {
        return;
    }
    WorkRange tokenRange = TilingByJobContext(common.tokenNum, context.job.jobIndex, context.job.totalJobs, 1U);
    if (tokenRange.count == 0U) {
        return;
    }
    int32_t jobLen = static_cast<int32_t>(tokenRange.count);
    int32_t jobOffset = static_cast<int32_t>(tokenRange.start);
    GlobalTensor<bfloat16_t> expandedX;
    expandedX.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(params.peermemInfo.combineSendPtr));
    GlobalTensor<bfloat16_t> output;
    output.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(params.y2GmAddr));

    constexpr TEventID outputBufferEvent = EVENT_ID0;
    SetFlag<HardEvent::MTE3_V>(outputBufferEvent);
    for (int32_t batchTokenOffset = 0; batchTokenOffset < jobLen; batchTokenOffset += bufferConfig.tokensPerBatch) {
        int32_t batchTokenCount = batchTokenOffset + bufferConfig.tokensPerBatch > jobLen ? jobLen - batchTokenOffset :
                                                                                            bufferConfig.tokensPerBatch;
        ProcessTokenUnpermuteBatch<CombineMode, TopkWeightsType, TopkWeightsPrefetch, Gmm1TileM>(
            context, common, params, scratch, bufferConfig, expandedX, output, jobOffset, batchTokenOffset,
            batchTokenCount, outputBufferEvent);
    }
    WaitFlag<HardEvent::MTE3_V>(outputBufferEvent);
}

} // namespace MegaMoeImpl

#endif // MEGA_MOE_UNPERMUTE_H
