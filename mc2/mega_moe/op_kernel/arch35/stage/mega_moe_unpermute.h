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
    // ==== wire combine（data-with-flag）扩展，wireBlockCount==0 表示旧布局 ====
    uint32_t wirePayloadBytes = 0;  // P：一条 combine 记录的真实 payload 字节数
    uint32_t wireBlockCount = 0;    // B = ceil(P / 480)
    uint32_t wireRowBytes = 0;      // W = B * 512
    uint32_t wireFlagScanBytes = 0; // Align32(topK * B * 32B)：flag gather 区大小（reduce tmp 同尺寸）
    // 去同步生效时为本轮 epoch（>0），unpermute 逐 token 等 flag；为 0 时按 wire 布局搬运但不等 flag
    //（该场景由保留的全卡同步保证数据已到）。
    int32_t epoch = 0;
};

struct TokenUnpermuteScratch {
    LocalTensor<bfloat16_t> dataResTensor;
    LocalTensor<float> dataResFp32Tensor;
    LocalTensor<float> topKWeightsTensor;
    LocalTensor<float> fp32ScaleTensor;
    LocalTensor<bfloat16_t> bf16ScaleTensor;
    LocalTensor<bfloat16_t> topKWeightsBf16Tensor;
    // wire flag 检查专用（排在既有 buffer 之后，host 侧已为其在 UB 预算中扣除预留）。
    LocalTensor<int32_t> wireFlagScanTensor;    // topK*B 个 32B flag 的 gather 目标
    LocalTensor<uint8_t> wireFlagReduceTensor;  // ReduceMax 的共享 tmp
    LocalTensor<int32_t> wireFlagResultTensor;  // reduce 结果（单值）
    LocalTensor<int32_t> wireFlagEpochTemplate; // 全 epoch 常量模板（XOR 判全等用，每 launch 填一次）
};

// wire flag 检查的事件号：输入 ring 最多占 EVENT_ID0..5、输出占 EVENT_ID0(MTE3_V)，7 号无冲突。
constexpr int32_t UNPERMUTE_WIRE_FLAG_EVENT_ID = 7;

/*
 * 等待一个 token 的全部 topK 条 combine 记录到齐：gather 该 token连续 topK*B 个块尾 32B flag
 * （等距 512B），全部 8*topK*B 个 int32 都等于本轮 epoch 才算到齐。
 * 判全等 = flags XOR 全 epoch 模板后 ReduceMax==0：单次 reduce、无求和混叠（通信域跨 case
 * 复用、shape 切换后 flag 位置可能是任意残留字节，逐位异或对任意残留都不产生假到齐）。
 */
__aicore__ inline void WaitCombineTokenWireFlags(const TokenUnpermuteConfig &context,
                                                 const MoeStageCommonConfig &common, TokenUnpermuteScratch &scratch,
                                                 const GlobalTensor<int32_t> &combineWinInt32, int32_t tokenIdx)
{
    uint32_t flagCount = common.topK * context.wireBlockCount;
    uint32_t flagElemCount = flagCount * static_cast<uint32_t>(INT32_PER_256B);
    uint64_t firstFlagInt32Index =
        (static_cast<uint64_t>(tokenIdx) * common.topK * context.wireRowBytes +
         static_cast<uint64_t>(COMBINE_WIRE_BLOCK_DATA_BYTES)) /
        sizeof(int32_t);
    // DataCopy（非 pad）：blockLen/stride 单位 32B。每块取 32B，跳过 480B payload。
    DataCopyParams gatherParams{static_cast<uint16_t>(flagCount), 1U,
                                static_cast<uint16_t>(COMBINE_WIRE_BLOCK_DATA_BYTES / ALIGN_32), 0U};
    const uint32_t flagShape[] = {1U, flagElemCount * 2U}; // 按 uint16 归约
    LocalTensor<uint16_t> scanU16 = scratch.wireFlagScanTensor.template ReinterpretCast<uint16_t>();
    LocalTensor<uint16_t> templU16 = scratch.wireFlagEpochTemplate.template ReinterpretCast<uint16_t>();
    LocalTensor<uint16_t> resultU16 = scratch.wireFlagResultTensor.template ReinterpretCast<uint16_t>();
    while (true) {
        // 自配对同步：上一轮（或上一 token）的 S 读与 V 归约先于本轮 scan/result 覆写。
        SyncFuncStatic<HardEvent::S_V, UNPERMUTE_WIRE_FLAG_EVENT_ID>();
        SyncFuncStatic<HardEvent::V_MTE2, UNPERMUTE_WIRE_FLAG_EVENT_ID>();
        DataCopy(scratch.wireFlagScanTensor, combineWinInt32[firstFlagInt32Index], gatherParams);
        SyncFuncStatic<HardEvent::MTE2_V, UNPERMUTE_WIRE_FLAG_EVENT_ID>();
        Xor(scanU16, scanU16, templU16, static_cast<int32_t>(flagElemCount * 2U));
        PipeBarrier<PIPE_V>();
        ReduceMax<uint16_t, AscendC::Pattern::Reduce::AR, true>(resultU16, scanU16, scratch.wireFlagReduceTensor,
                                                                flagShape, true);
        SyncFuncStatic<HardEvent::V_S, UNPERMUTE_WIRE_FLAG_EVENT_ID>();
        if (resultU16.GetValue(0) == 0U) {
            return;
        }
        int64_t waitStart = AscendC::GetSystemCycle();
        while (AscendC::GetSystemCycle() - waitStart < 100) {
        }
    }
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

// 从 wire 布局（480B payload + 32B flag 交错）把一条记录的 payload 剥出为连续 UB 数据。
// 实测取舍：480B 分段 strided MTE2 读优于「整行连续读+V 压缩」——unpermute 段 V 管线本就被
// Cast/加权累加占满，V 压缩反而抢占关键资源（连续读+V 压缩实测 8K 慢 ~460us，勿再试）。
__aicore__ inline void LoadWireCombinePayload(const TokenUnpermuteConfig &context,
                                              const GlobalTensor<bfloat16_t> &expandedX, uint64_t expertInputIndex,
                                              LocalTensor<bfloat16_t> &dataInBf16)
{
    constexpr uint32_t wireDataBytes = static_cast<uint32_t>(COMBINE_WIRE_BLOCK_DATA_BYTES);
    constexpr uint32_t wireBlockBytes = static_cast<uint32_t>(COMBINE_WIRE_BLOCK_BYTES);
    uint32_t fullBlockCount = context.wireBlockCount - 1U;
    uint32_t tailBytes = context.wirePayloadBytes - fullBlockCount * wireDataBytes;
    uint64_t rowElemBase = expertInputIndex * context.wireRowBytes / sizeof(bfloat16_t);
    DataCopyPadExtParams<bfloat16_t> padParams{false, 0U, 0U, 0U};
    if (fullBlockCount > 0U) {
        // GM 侧每 480B payload 后跳 32B flag（srcStride 单位字节），UB 侧连续。
        DataCopyExtParams fullParams{static_cast<uint16_t>(fullBlockCount), wireDataBytes,
                                     static_cast<uint32_t>(ALIGN_32), 0U, 0U};
        DataCopyPad(dataInBf16, expandedX[rowElemBase], fullParams, padParams);
    }
    DataCopyExtParams tailParams{1U, tailBytes, 0U, 0U, 0U};
    DataCopyPad(dataInBf16[fullBlockCount * wireDataBytes / sizeof(bfloat16_t)],
                expandedX[rowElemBase + static_cast<uint64_t>(fullBlockCount) * wireBlockBytes / sizeof(bfloat16_t)],
                tailParams, padParams);
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
        if (context.wireBlockCount != 0U) {
            LoadWireCombinePayload(context, expandedX, expertInputIndex, dataInBf16);
        } else {
            DataCopy(dataInBf16, expandedX[expertInputIndex * common.tokenHiddenDim], common.tokenHiddenDim);
        }
        SetFlag<HardEvent::MTE2_V>(event);
        WaitFlag<HardEvent::MTE2_V>(event);
        Cast(dataInFp32, dataInBf16, RoundMode::CAST_NONE, common.tokenHiddenDim);
    } else {
        if (context.wireBlockCount != 0U) {
            LoadWireCombinePayload(context, expandedX, expertInputIndex, dataInBf16);
        } else {
            uint32_t quantTokenElementCount = context.quantTokenSizeBytes / sizeof(bfloat16_t);
            DataCopy(dataInBf16, expandedX[expertInputIndex * quantTokenElementCount], quantTokenElementCount);
        }
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
                                                    const MoeStageCommonConfig &common, TokenUnpermuteScratch &scratch,
                                                    int32_t tokenIdx, int32_t localIdx,
                                                    const GlobalTensor<bfloat16_t> &expandedX,
                                                    const GlobalTensor<int32_t> &combineWinInt32,
                                                    const MegaMoeUnpermuteBufferConfig &bufferConfig)
{
    // 去同步生效时（epoch>0）先等本 token 全部 topK 条记录的 data-with-flag 到齐再消费。
    if (context.epoch != 0) {
        WaitCombineTokenWireFlags(context, common, scratch, combineWinInt32, tokenIdx);
    }
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
    Add(scratch.dataResFp32Tensor, scratch.dataResFp32Tensor, dataInFp32, common.tokenHiddenDim);
    PipeBarrier<PIPE_V>();
    SetFlag<HardEvent::V_MTE2>(event);
}

template <uint8_t CombineMode, typename TopkWeightsType, bool TopkWeightsPrefetch, uint32_t Gmm1TileM>
__aicore__ inline void ProcessTokenUnpermuteBatch(
    const TokenUnpermuteConfig &context, const MoeStageCommonConfig &common, const Params &params,
    TokenUnpermuteScratch &scratch, const MegaMoeUnpermuteBufferConfig &bufferConfig,
    const GlobalTensor<bfloat16_t> &expandedX, const GlobalTensor<int32_t> &combineWinInt32,
    GlobalTensor<bfloat16_t> &output, int32_t jobOffset, int32_t batchTokenOffset, int32_t batchTokenCount,
    TEventID outputBufferEvent)
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
        AccumulateMoeExpertsForToken<CombineMode, TopkWeightsPrefetch>(context, common, scratch, tokenIdx, localIdx,
                                                                       expandedX, combineWinInt32, bufferConfig);
        for (uint32_t sharedExpertIdx = 0; sharedExpertIdx < common.sharedExpertNum; ++sharedExpertIdx) {
            AccumulateSharedExpertForToken<Gmm1TileM>(common, params, scratch, tokenIdx, localIdx, sharedExpertIdx,
                                                      bufferConfig);
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
        tempAddr += fp32ScaleBufAlign;
    }
    if (context.wireBlockCount != 0U) {
        // flag gather 区 + reduce tmp（同尺寸上界）+ epoch 模板区 + 单值结果；
        // wireFlagScanBytes 由 wave Init 按 topK*B*32B 算好，host 侧按同一公式扣除 UB 预留（×3+32）。
        uint32_t flagScanBytes = context.wireFlagScanBytes;
        scratch.wireFlagScanTensor = LocalTensor<int32_t>(TPosition::VECCALC, tempAddr, flagScanBytes / sizeof(int32_t));
        tempAddr += flagScanBytes;
        scratch.wireFlagReduceTensor = LocalTensor<uint8_t>(TPosition::VECCALC, tempAddr, flagScanBytes);
        tempAddr += flagScanBytes;
        scratch.wireFlagEpochTemplate =
            LocalTensor<int32_t>(TPosition::VECCALC, tempAddr, flagScanBytes / sizeof(int32_t));
        tempAddr += flagScanBytes;
        scratch.wireFlagResultTensor =
            LocalTensor<int32_t>(TPosition::VECCALC, tempAddr, static_cast<uint32_t>(ALIGN_32) / sizeof(int32_t));
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
    GlobalTensor<int32_t> combineWinInt32;
    combineWinInt32.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(params.peermemInfo.combineSendPtr));
    GlobalTensor<bfloat16_t> output;
    output.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(params.y2GmAddr));

    if (context.epoch != 0) {
        // XOR 判全等的 epoch 常量模板，每 launch 填一次；与后续 MTE2/归约建立一次性顺序。
        Duplicate<int32_t>(scratch.wireFlagEpochTemplate, context.epoch,
                           static_cast<int32_t>(context.wireFlagScanBytes / sizeof(int32_t)));
        PipeBarrier<PIPE_V>();
    }
    constexpr TEventID outputBufferEvent = EVENT_ID0;
    SetFlag<HardEvent::MTE3_V>(outputBufferEvent);
    for (int32_t batchTokenOffset = 0; batchTokenOffset < jobLen; batchTokenOffset += bufferConfig.tokensPerBatch) {
        int32_t batchTokenCount = batchTokenOffset + bufferConfig.tokensPerBatch > jobLen ? jobLen - batchTokenOffset :
                                                                                            bufferConfig.tokensPerBatch;
        ProcessTokenUnpermuteBatch<CombineMode, TopkWeightsType, TopkWeightsPrefetch, Gmm1TileM>(
            context, common, params, scratch, bufferConfig, expandedX, combineWinInt32, output, jobOffset,
            batchTokenOffset, batchTokenCount, outputBufferEvent);
    }
    WaitFlag<HardEvent::MTE3_V>(outputBufferEvent);
}

} // namespace MegaMoeImpl

#endif // MEGA_MOE_UNPERMUTE_H
