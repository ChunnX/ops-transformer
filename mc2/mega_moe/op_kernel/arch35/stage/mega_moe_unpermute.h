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
    // dataResTensor 上首个输入槽的元素偏移：逐条读布局 = 单条槽宽（槽 0 为输出暂存且与输入槽等宽），
    // token 批读布局 = 输出暂存宽度（2H，槽 0 独立小槽，输入槽为 K 条记录宽）。
    uint32_t dataResInputBaseElementCount;
};

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

/*
 * token 批读路径的搬运发射（与 AccumulateMoeExpertsTokenBatch 配对）：接收窗口同一 token 的
 * topK 条记录物理连续（行公式 tokenIdx*topK+topkIdx），等待槽位空闲后一次 strided DataCopyPad
 * 发射整段，仅置 MTE2_V 到达标志、立即返回。发射与消费分离后，前序 token 消费（S/V 队列）
 * 期间 MTE2 即可并行搬运后续 token。槽位/事件与消费端一致绑定 token 序号模批槽数。
 * 记录在槽内按 256B 对齐步进落位（DeQuantMxFp8 的 SIMT 访问要求记录基址 256 对齐，逐条读
 * 路径每条记录落槽基址天然满足）：GM 侧记录物理连续（srcStride=0），UB 侧记录间垫对齐死区
 * （死区恒为 32 的倍数，dstStride 按 UB 侧单位 dataBlock=32B 换算为块数）。
 */
__aicore__ inline void IssueTokenBatchCopy(const TokenUnpermuteConfig &context, const MoeStageCommonConfig &common,
                                           TokenUnpermuteScratch &scratch, int32_t tokenIdx, uint32_t tokenSlotIdx,
                                           const GlobalTensor<bfloat16_t> &expandedX,
                                           const MegaMoeUnpermuteBufferConfig &bufferConfig)
{
    TEventID event = static_cast<TEventID>(tokenSlotIdx);
    LocalTensor<bfloat16_t> batchUb =
        scratch.dataResTensor[scratch.dataResInputBaseElementCount + tokenSlotIdx * bufferConfig.bf16SlotElementCount];
    uint32_t recordElementCount = context.quantTokenSizeBytes / sizeof(bfloat16_t);
    // DataCopyPad 的 stride 单位按操作数位置区分：GM 侧为字节，UB（VECIN/VECOUT）侧为 dataBlock(32B)。
    // 本拷贝目的在 UB，块间隔（对齐步进 − 记录字节数）恒为 32 的倍数，整除换算为 32B 块数下发。
    uint32_t recordGapBlocks =
        (bufferConfig.tokenBatchRecordStrideElementCount * static_cast<uint32_t>(sizeof(bfloat16_t)) -
         context.quantTokenSizeBytes) /
        static_cast<uint32_t>(ALIGN_32);
    DataCopyExtParams batchCopyParams{static_cast<uint16_t>(common.topK), context.quantTokenSizeBytes, 0U,
                                      recordGapBlocks, 0U};
    DataCopyPadExtParams<bfloat16_t> batchCopyPad{false, 0U, 0U, 0U};
    WaitFlag<HardEvent::V_MTE2>(event);
    DataCopyPad(batchUb, expandedX[static_cast<uint64_t>(tokenIdx) * common.topK * recordElementCount], batchCopyParams,
                batchCopyPad);
    SetFlag<HardEvent::MTE2_V>(event);
}

/*
 * token 批读路径的消费侧（useTokenBatchRead 布局专用）：搬运已由 IssueTokenBatchCopy 提前发射，
 * 此处单次 MTE2_V 等待覆盖全部 K 条记录，随后在槽内按 256B 对齐步进逐条 DeQuant/累加（每条
 * 记录基址与逐条读路径的槽基址同等对齐）。每条记录的处理顺序与逐条读路径完全一致，浮点累加
 * 序不变。末尾 V_MTE2 释放槽位，供共享专家行暂存（批读布局同槽复用）及该槽下一个 token 的
 * 搬运发射。
 */
template <uint8_t CombineMode, bool TopkWeightsPrefetch>
__aicore__ inline void AccumulateMoeExpertsTokenBatch(const MoeStageCommonConfig &common,
                                                      TokenUnpermuteScratch &scratch, int32_t localIdx,
                                                      const MegaMoeUnpermuteBufferConfig &bufferConfig)
{
    uint32_t tokenSlotIdx = static_cast<uint32_t>(localIdx) % static_cast<uint32_t>(bufferConfig.inputBufferCount);
    TEventID event = static_cast<TEventID>(tokenSlotIdx);
    LocalTensor<bfloat16_t> batchUb =
        scratch.dataResTensor[scratch.dataResInputBaseElementCount + tokenSlotIdx * bufferConfig.bf16SlotElementCount];
    LocalTensor<float> workFp32 = scratch.dataResFp32Tensor[bufferConfig.fp32SlotElementCount];
    WaitFlag<HardEvent::MTE2_V>(event);
    uint32_t nScale = Ops::Base::CeilDiv(common.tokenHiddenDim, static_cast<uint32_t>(MXFP_SCALE_GROUP_NUM));
    using Fp8Type = typename std::conditional<CombineMode == MXFP8_E4M3_COMM_QUANT, fp8_e4m3fn_t, fp8_e5m2_t>::type;
    for (int32_t expertIdx = 0; expertIdx < static_cast<int32_t>(common.topK); ++expertIdx) {
        LocalTensor<bfloat16_t> recordUb =
            batchUb[static_cast<uint32_t>(expertIdx) * bufferConfig.tokenBatchRecordStrideElementCount];
        Mxfp8::DeQuantMxFp8<Fp8Type, bfloat16_t>(recordUb, workFp32, scratch.bf16ScaleTensor, scratch.fp32ScaleTensor,
                                                 nScale, common.tokenHiddenDim);
        SetFlag<HardEvent::S_V>(event);
        WaitFlag<HardEvent::S_V>(event);
        PipeBarrier<PIPE_V>();
        if (expertIdx == 0) {
            if constexpr (TopkWeightsPrefetch) {
                /*
                 * 权重已在 dispatch 记录内前乘，首条为纯拷贝；改用 Muls(1.0) 而非 DataCopy：
                 * workFp32 是单槽复用，UB->UB DataCopy 若走 MTE 引擎会与下一条 DeQuant（V 引擎
                 * 写同一 workFp32）跨引擎并发；Muls 是 V 指令，V 队列按序执行天然消除竞争，
                 * 且 fp32 乘 1.0 逐位精确（含 ±0 保号）。逐条读路径无此问题（各条独立槽）。
                 */
                Muls(scratch.dataResFp32Tensor, workFp32, 1.0f, common.tokenHiddenDim);
            } else {
                float expertScale = scratch.topKWeightsTensor.GetValue(localIdx * common.topK + expertIdx);
                Muls(scratch.dataResFp32Tensor, workFp32, expertScale, common.tokenHiddenDim);
            }
        } else {
            if constexpr (!TopkWeightsPrefetch) {
                float expertScale = scratch.topKWeightsTensor.GetValue(localIdx * common.topK + expertIdx);
                Muls(workFp32, workFp32, expertScale, common.tokenHiddenDim);
                PipeBarrier<PIPE_V>();
            }
            Add(scratch.dataResFp32Tensor, scratch.dataResFp32Tensor, workFp32, common.tokenHiddenDim);
            PipeBarrier<PIPE_V>();
        }
    }
    SetFlag<HardEvent::V_MTE2>(event);
}

// 累加一个 token 对应的全部 MoE 专家输入。
template <uint8_t CombineMode, bool TopkWeightsPrefetch>
__aicore__ inline void AccumulateMoeExpertsForToken(const TokenUnpermuteConfig &context,
                                                    const MoeStageCommonConfig &common, TokenUnpermuteScratch &scratch,
                                                    int32_t tokenIdx, int32_t localIdx,
                                                    const GlobalTensor<bfloat16_t> &expandedX,
                                                    const MegaMoeUnpermuteBufferConfig &bufferConfig)
{
    if constexpr (CombineMode != COMBINE_NO_QUANT) {
        if (bufferConfig.useTokenBatchRead != 0U) {
            AccumulateMoeExpertsTokenBatch<CombineMode, TopkWeightsPrefetch>(common, scratch, localIdx, bufferConfig);
            return;
        }
    }
    for (int32_t expertIdx = 0; expertIdx < static_cast<int32_t>(common.topK); ++expertIdx) {
        int32_t accumulationItemIdx = localIdx * static_cast<int32_t>(common.topK + common.sharedExpertNum) + expertIdx;
        int32_t inputBufferIdx = accumulationItemIdx % bufferConfig.inputBufferCount;
        TEventID event = static_cast<TEventID>(inputBufferIdx);
        LocalTensor<bfloat16_t> dataInBf16 = scratch.dataResTensor[scratch.dataResInputBaseElementCount +
                                                                   inputBufferIdx * bufferConfig.bf16SlotElementCount];
        LocalTensor<float> dataInFp32 =
            scratch.dataResFp32Tensor[(inputBufferIdx + 1) * bufferConfig.fp32SlotElementCount];
        uint64_t expertInputIndex = static_cast<uint64_t>(tokenIdx) * common.topK + expertIdx;
        LoadMoeExpertInput<CombineMode>(context, common, scratch, expandedX, expertInputIndex, event, dataInBf16,
                                        dataInFp32);
        SetFlag<HardEvent::S_V>(event);
        WaitFlag<HardEvent::S_V>(event);
        PipeBarrier<PIPE_V>();
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
    // 批读布局复用当前 token 的批槽（MoE 消费末尾已置 V_MTE2 释放）：发射提前后其余批槽可能
    // 持有在飞 token 数据，不能再按 accumulationItemIdx 取槽覆盖。逐条读布局维持原映射。
    int32_t inputBufferIdx = bufferConfig.useTokenBatchRead != 0U ? localIdx % bufferConfig.inputBufferCount :
                                                                    accumulationItemIdx % bufferConfig.inputBufferCount;
    TEventID event = static_cast<TEventID>(inputBufferIdx);
    LocalTensor<bfloat16_t> dataInBf16 =
        scratch
            .dataResTensor[scratch.dataResInputBaseElementCount + inputBufferIdx * bufferConfig.bf16SlotElementCount];
    LocalTensor<float> dataInFp32 =
        bufferConfig.useTokenBatchRead != 0U ?
            scratch.dataResFp32Tensor[bufferConfig.fp32SlotElementCount] :
            scratch.dataResFp32Tensor[(inputBufferIdx + 1) * bufferConfig.fp32SlotElementCount];
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
    /*
     * token 批读布局的发射-消费分离流水（CO-5b）：prime 先把前 inputBufferCount 个 token 的
     * 整段搬运全部发射；steady 中第 i 个 token 消费时其数据早已在飞/到达，消费期间（S/V 队列）
     * MTE2 并行搬运第 i+N 个 token。补发排在共享专家累加之后（本 token 批槽经其末尾 V_MTE2
     * 释放，且其余批槽可能持有在飞数据）、输出之前（输出暂存区与批槽不相交，MTE2 可与
     * 输出等待/Cast 重叠）。逐条读布局不经过本编排，行为不变。
     */
    bool useTokenBatchRead = false;
    if constexpr (CombineMode != COMBINE_NO_QUANT) {
        useTokenBatchRead = bufferConfig.useTokenBatchRead != 0U;
    }
    if (useTokenBatchRead) {
        int32_t primeTokenCount =
            bufferConfig.inputBufferCount < batchTokenCount ? bufferConfig.inputBufferCount : batchTokenCount;
        for (int32_t primeIdx = 0; primeIdx < primeTokenCount; ++primeIdx) {
            IssueTokenBatchCopy(context, common, scratch, jobOffset + batchTokenOffset + primeIdx,
                                static_cast<uint32_t>(primeIdx) % static_cast<uint32_t>(bufferConfig.inputBufferCount),
                                expandedX, bufferConfig);
        }
    }
    for (int32_t localIdx = 0; localIdx < batchTokenCount; ++localIdx) {
        int32_t tokenIdx = jobOffset + batchTokenOffset + localIdx;
        AccumulateMoeExpertsForToken<CombineMode, TopkWeightsPrefetch>(context, common, scratch, tokenIdx, localIdx,
                                                                       expandedX, bufferConfig);
        for (uint32_t sharedExpertIdx = 0; sharedExpertIdx < common.sharedExpertNum; ++sharedExpertIdx) {
            AccumulateSharedExpertForToken<Gmm1TileM>(common, params, scratch, tokenIdx, localIdx, sharedExpertIdx,
                                                      bufferConfig);
        }
        if (useTokenBatchRead && localIdx + bufferConfig.inputBufferCount < batchTokenCount) {
            IssueTokenBatchCopy(context, common, scratch,
                                jobOffset + batchTokenOffset + localIdx + bufferConfig.inputBufferCount,
                                static_cast<uint32_t>(localIdx) % static_cast<uint32_t>(bufferConfig.inputBufferCount),
                                expandedX, bufferConfig);
        }
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
 * UB 从地址 0 起按下列顺序连续排布（每段起点=上一段末尾）。
 * 逐条读布局（N = inputBufferCount）：
 *   [dataRes    ] (N+1) 个 bf16 槽 × bf16SlotElementCount —— 累加结果 bf16 输出
 *   [dataResFp32] (N+1) 个 fp32 槽 × fp32SlotElementCount —— fp32 累加中间态
 * token 批读布局（量化，N = inputBufferCount 为 token 批槽数，槽容量 = topK×256 对齐记录步进，
 * 记录间垫对齐死区）：
 *   [dataRes    ] 1 个输出暂存 bf16 槽（2H 按 256 对齐）+ N 个 token 批槽 × bf16SlotElementCount
 *   [dataResFp32] 累加器 + 单工作槽，共 2 个 fp32 槽 × fp32SlotElementCount
 * 两种布局共用尾部：
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

    const bool useTokenBatchRead = CombineQuantMode != COMBINE_NO_QUANT && bufferConfig.useTokenBatchRead != 0U;
    uint32_t bf16SlotBytes = bufferConfig.bf16SlotElementCount * sizeof(bfloat16_t);
    uint32_t fp32SlotBytes = bufferConfig.fp32SlotElementCount * sizeof(float);
    uint32_t dataResBufBytes;
    uint32_t dataResFp32BufBytes;
    if (useTokenBatchRead) {
        // 输出暂存按 256B 对齐（与 host TryCalcTokenBatchReadConfig 同口径）：槽步进 = topK×256
        // 对齐记录步进，故各批槽基址及槽内记录基址均 256 对齐，满足 DeQuantMxFp8 的 SIMT 访问要求。
        uint32_t bf16OutputSlotBytes = static_cast<uint32_t>(Ops::Base::CeilAlign(
            static_cast<uint64_t>(tokenHiddenDim) * sizeof(bfloat16_t), static_cast<uint64_t>(ALIGN_256)));
        scratch.dataResInputBaseElementCount = bf16OutputSlotBytes / sizeof(bfloat16_t);
        dataResBufBytes = bf16OutputSlotBytes + static_cast<uint32_t>(bufferConfig.inputBufferCount) * bf16SlotBytes;
        dataResFp32BufBytes = 2U * fp32SlotBytes;
    } else {
        scratch.dataResInputBaseElementCount = bufferConfig.bf16SlotElementCount;
        dataResBufBytes = (bufferConfig.inputBufferCount + 1) * bf16SlotBytes;
        dataResFp32BufBytes = (bufferConfig.inputBufferCount + 1) * fp32SlotBytes;
    }
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

// 执行当前 token 分区的完整 Unpermute 流水。
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
