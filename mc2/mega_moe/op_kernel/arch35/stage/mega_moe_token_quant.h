/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MEGA_MOE_TOKEN_QUANT_H
#define MEGA_MOE_TOKEN_QUANT_H

#include "../common/mega_moe_utils.h"
#if __has_include("../../../common/quantize_functions.h")
#include "../../../common/quantize_functions.h"
#else
#include "../../../../common/op_kernel/quantize_functions.h"
#endif

namespace MegaMoeImpl {

using namespace AscendC;

struct QuantProcessConfig {
    uint32_t quantTokenAlignBytes;
    uint32_t quantScaleAlignBytes;
    uint32_t quantTokenScaleAlignBytes;
    uint32_t quantScaleValidCountPerToken;
};

/*
 * 计算量化通信记录的对齐布局（普通与 wave 编排模板共用）。
 * 记录布局 = Align256(token 数据) + Align32(scale) + prefetch 时附加 Align32(topk 权重)，
 * 与 host CalcDispatchBufferConfig 的 copyBufferBytes 契约恒相等。
 */
template <typename ActivationType, typename QuantScaleOutType, bool TopkWeightsPrefetch, uint32_t AElemsPerByte>
__aicore__ inline QuantProcessConfig CreateQuantProcessConfig(uint32_t tokenHiddenDim, const Params &params)
{
    uint32_t quantScaleValidCountPerToken = Ops::Base::CeilDiv(tokenHiddenDim, static_cast<uint32_t>(ALIGN_32));
    uint32_t quantTokenAlignBytes =
        Ops::Base::CeilAlign(tokenHiddenDim / AElemsPerByte, static_cast<uint32_t>(ALIGN_256)) * sizeof(ActivationType);
    uint32_t quantScaleAlignBytes =
        Ops::Base::CeilAlign(quantScaleValidCountPerToken * static_cast<uint32_t>(sizeof(QuantScaleOutType)),
                             static_cast<uint32_t>(ALIGN_32));
    uint32_t quantTokenScaleAlignBytes = quantTokenAlignBytes + quantScaleAlignBytes;
    if constexpr (TopkWeightsPrefetch) {
        uint32_t weightAlignBytes = Ops::Base::CeilAlign(static_cast<uint32_t>(params.tilingData->topK * sizeof(float)),
                                                         static_cast<uint32_t>(ALIGN_32));
        quantTokenScaleAlignBytes += weightAlignBytes;
    }
    return {quantTokenAlignBytes, quantScaleAlignBytes, quantTokenScaleAlignBytes, quantScaleValidCountPerToken};
}

template <typename ActivationType>
struct QuantProcessScratch {
    LocalTensor<bfloat16_t> xInTensor0;
    LocalTensor<bfloat16_t> xInTensor1;
    LocalTensor<ActivationType> xOutTensor0;
    LocalTensor<ActivationType> xOutTensor1;
    LocalTensor<uint16_t> mxTempTensor;
    // 预量化路径只使用这两个独立搬运 slot；动态量化仍使用上面的旧布局。
    LocalTensor<uint8_t> preQuantStage0;
    LocalTensor<uint8_t> preQuantStage1;
};

template <typename TopkWeightsType, typename ActivationType>
__aicore__ inline void PrefetchTopkWeights(GM_ADDR tokenTopkWeightsAddr, uint32_t topK,
                                           const QuantProcessConfig &config,
                                           QuantProcessScratch<ActivationType> &scratch,
                                           const LocalTensor<ActivationType> &xOutTensor, TEventID event)
{
    GlobalTensor<TopkWeightsType> weightGm;
    weightGm.SetGlobalBuffer(reinterpret_cast<__gm__ TopkWeightsType *>(tokenTopkWeightsAddr));
    uint32_t weightOffsetInUb = config.quantTokenAlignBytes + config.quantScaleAlignBytes;
    if constexpr (Std::IsSame<TopkWeightsType, bfloat16_t>::value) {
        LocalTensor<TopkWeightsType> weightBf16Tmp = scratch.mxTempTensor.template ReinterpretCast<TopkWeightsType>();
        DataCopyPad(weightBf16Tmp, weightGm, {1U, static_cast<uint32_t>(topK * sizeof(TopkWeightsType)), 0U, 0U, 0U},
                    {false, 0U, 0U, 0U});
        SetFlag<AscendC::HardEvent::MTE2_V>(event);
        WaitFlag<AscendC::HardEvent::MTE2_V>(event);
        LocalTensor<float> weightFp32Ub = xOutTensor[weightOffsetInUb].template ReinterpretCast<float>();
        Cast(weightFp32Ub, weightBf16Tmp, AscendC::RoundMode::CAST_NONE, topK);
        PipeBarrier<PIPE_V>();
    } else {
        LocalTensor<TopkWeightsType> weightUb =
            xOutTensor[weightOffsetInUb].template ReinterpretCast<TopkWeightsType>();
        DataCopyPad(weightUb, weightGm, {1U, static_cast<uint32_t>(topK * sizeof(TopkWeightsType)), 0U, 0U, 0U},
                    {false, 0U, 0U, 0U});
        SetFlag<AscendC::HardEvent::MTE2_V>(event);
        WaitFlag<AscendC::HardEvent::MTE2_V>(event);
    }
}

template <int32_t QuantMode, typename QuantOutType, typename ActivationType>
__aicore__ inline void QuantizeTokenInUb(const LocalTensor<bfloat16_t> &xInTensor,
                                         const LocalTensor<ActivationType> &xOutTensor,
                                         const LocalTensor<uint16_t> &mxTempTensor, const QuantProcessConfig &config,
                                         uint32_t hiddenDim)
{
    __ubuf__ uint16_t *maxExpAddr = reinterpret_cast<__ubuf__ uint16_t *>(mxTempTensor.GetPhyAddr());
    __ubuf__ uint16_t *halfScaleAddr = reinterpret_cast<__ubuf__ uint16_t *>(
        mxTempTensor[Ops::Base::CeilAlign(config.quantScaleValidCountPerToken, static_cast<uint32_t>(ALIGN_32))]
            .GetPhyAddr());
    __ubuf__ bfloat16_t *srcAddr = reinterpret_cast<__ubuf__ bfloat16_t *>(xInTensor.GetPhyAddr());
    __ubuf__ int8_t *outDataAddr = reinterpret_cast<__ubuf__ int8_t *>(xOutTensor.GetPhyAddr());
    __ubuf__ uint16_t *mxScaleAddr =
        reinterpret_cast<__ubuf__ uint16_t *>(xOutTensor[config.quantTokenAlignBytes].GetPhyAddr());

    Quant::ComputeMaxExp(srcAddr, maxExpAddr, hiddenDim);
    Quant::ComputeScale<QuantOutType>(maxExpAddr, mxScaleAddr, halfScaleAddr, config.quantScaleValidCountPerToken);
    if constexpr (QuantMode == E2M1_QUANT) {
        Quant::ComputeFp4Data<bfloat16_t, QuantOutType, AscendC::RoundMode::CAST_TRUNC, AscendC::RoundMode::CAST_RINT>(
            srcAddr, halfScaleAddr, outDataAddr, hiddenDim);
    } else {
        Quant::ComputeFp8Data<bfloat16_t, QuantOutType, AscendC::RoundMode::CAST_TRUNC, AscendC::RoundMode::CAST_RINT>(
            srcAddr, halfScaleAddr, outDataAddr, hiddenDim);
    }
}

// 量化一个逻辑 AIV 任务负责的本卡 token，并按逐 token 交织布局写入量化数据与 scale。
template <int32_t QuantMode, typename QuantOutType, typename ActivationType, typename TopkWeightsType,
          bool TopkWeightsPrefetch>
__aicore__ inline void QuantizeLocalTokens(const AivJobContext &job, const MoeStageCommonConfig &common,
                                           const Params &params, const QuantProcessConfig &config, GM_ADDR outputAddr,
                                           QuantProcessScratch<ActivationType> &scratch)
{
    if constexpr (g_coreType == AIC) {
        return;
    }
    WorkRange tokenRange = TilingByJobContext(common.tokenNum, job.jobIndex, job.totalJobs, 1U);
    if (tokenRange.count == 0U) {
        return;
    }
    GlobalTensor<uint8_t> output;
    output.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(outputAddr));
    uint32_t hiddenDim = common.tokenHiddenDim;
    GlobalTensor<bfloat16_t> srcGlobalTensor;
    srcGlobalTensor.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(params.aGmAddr) +
                                    static_cast<uint64_t>(tokenRange.start) * hiddenDim);
    // 量化 scratch（mxTemp/xOut0/xOut1/xIn0/xIn1）的跨 launch 残留清零由各编排的
    // SendAndQuantBuffInit 在分配处一次性完成（span 清零，与布局同源），见其注释。
    SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
    for (uint32_t index = 0; index < tokenRange.count; ++index) {
        bool useFirstBuffer = index % DOUBLE_BUFFER == 0;
        auto event = useFirstBuffer ? EVENT_ID0 : EVENT_ID1;
        auto xInTensor = useFirstBuffer ? scratch.xInTensor0 : scratch.xInTensor1;
        auto xOutTensor = useFirstBuffer ? scratch.xOutTensor0 : scratch.xOutTensor1;
        WaitFlag<AscendC::HardEvent::MTE3_MTE2>(event);
        DataCopyPad(xInTensor, srcGlobalTensor[static_cast<uint64_t>(index) * hiddenDim],
                    {1U, static_cast<uint16_t>(hiddenDim * sizeof(bfloat16_t)), 0U, 0U}, {true, 0, 0, 0});
        uint32_t tokenIndex = tokenRange.start + index;
        if constexpr (TopkWeightsPrefetch) {
            GM_ADDR tokenTopkWeightsAddr =
                params.probsGmAddr + static_cast<uint64_t>(tokenIndex) * common.topK * sizeof(TopkWeightsType);
            PrefetchTopkWeights<TopkWeightsType>(tokenTopkWeightsAddr, common.topK, config, scratch, xOutTensor, event);
        } else {
            SetFlag<AscendC::HardEvent::MTE2_V>(event);
            WaitFlag<AscendC::HardEvent::MTE2_V>(event);
        }
        QuantizeTokenInUb<QuantMode, QuantOutType>(xInTensor, xOutTensor, scratch.mxTempTensor, config, hiddenDim);
        SetFlag<AscendC::HardEvent::V_MTE3>(event);
        WaitFlag<AscendC::HardEvent::V_MTE3>(event);
        auto xOutBytesTensor = xOutTensor.template ReinterpretCast<uint8_t>();
        DataCopyPad(output[static_cast<uint64_t>(tokenIndex) * config.quantTokenScaleAlignBytes], xOutBytesTensor,
                    {1U, config.quantTokenScaleAlignBytes, 0U, 0U, 0U});
        SetFlag<AscendC::HardEvent::MTE3_MTE2>(event);
    }
    WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
}

// 聚合预量化搬运的 GM 视图和参数，不负责申请 GM 或 UB 空间。
struct PreQuantizedCopyContext {
    uint32_t dataValidBytes;
    uint32_t scaleValidBytes;
    uint32_t dataRowBytes;
    uint32_t scaleRowBytes;
    GlobalTensor<uint8_t> dataSrcGlobalTensor;
    GlobalTensor<uint8_t> scaleSrcGlobalTensor;
    GlobalTensor<uint8_t> tokenScaleDstGlobalTensor;
};

// GM -> UB。对齐时合并成一次普通 DataCopy，其他情况用 DataCopyPad 把行补到 32B。
__aicore__ inline void PreQuantCopyIn(LocalTensor<uint8_t> dst, GlobalTensor<uint8_t> src,
                                      uint32_t tokenCount, uint32_t validBytes, uint32_t rowBytes)
{
    if (tokenCount == 0U || validBytes == 0U) {
        return;
    }
    constexpr uint32_t blockBytes = static_cast<uint32_t>(ALIGN_32);
    uint32_t alignedBytes = Ops::Base::CeilAlign(validBytes, blockBytes);
    bool srcAligned = (reinterpret_cast<uint64_t>(src.GetPhyAddr()) & (blockBytes - 1U)) == 0U;
    bool dstAligned = (reinterpret_cast<uint64_t>(dst.GetPhyAddr()) & (blockBytes - 1U)) == 0U;
    if (validBytes % blockBytes == 0U && srcAligned && dstAligned) {
        if (rowBytes == validBytes) {
            DataCopy(dst, src, tokenCount * validBytes);
        } else {
            DataCopyExtParams params{static_cast<uint16_t>(tokenCount), validBytes, 0U,
                                     static_cast<uint32_t>((rowBytes - validBytes) / blockBytes), 0U};
            DataCopyPadExtParams<uint8_t> padParams{true, 0U, 0U, 0U};
            DataCopyPad(dst, src, params, padParams);
        }
        return;
    }

    DataCopyExtParams params{static_cast<uint16_t>(tokenCount), validBytes, 0U,
                             static_cast<uint32_t>((rowBytes - alignedBytes) / blockBytes), 0U};
    DataCopyPadExtParams<uint8_t> padParams{true, 0U, 0U, 0U};
    DataCopyPad(dst, src, params, padParams);
}

// UB -> GM。srcStride 走 UB 行间距，dstStride 走通信 record 的间距。
__aicore__ inline void PreQuantScatter(GlobalTensor<uint8_t> dst, LocalTensor<uint8_t> src,
                                       uint32_t tokenCount, uint32_t validBytes, uint32_t rowBytes,
                                       uint32_t recordBytes)
{
    if (tokenCount == 0U || validBytes == 0U) {
        return;
    }
    constexpr uint32_t blockBytes = static_cast<uint32_t>(ALIGN_32);
    uint32_t alignedBytes = Ops::Base::CeilAlign(validBytes, blockBytes);
    bool srcAligned = (reinterpret_cast<uint64_t>(src.GetPhyAddr()) & (blockBytes - 1U)) == 0U;
    bool dstAligned = (reinterpret_cast<uint64_t>(dst.GetPhyAddr()) & (blockBytes - 1U)) == 0U;
    if (validBytes % blockBytes == 0U && srcAligned && dstAligned) {
        DataCopyParams params{static_cast<uint16_t>(tokenCount), static_cast<uint16_t>(validBytes / blockBytes),
                              static_cast<uint16_t>((rowBytes - validBytes) / blockBytes),
                              static_cast<uint16_t>((recordBytes - validBytes) / blockBytes)};
        DataCopy(dst, src, params);
        return;
    }

    DataCopyExtParams params{static_cast<uint16_t>(tokenCount), validBytes,
                             static_cast<uint32_t>((rowBytes - alignedBytes) / blockBytes),
                             recordBytes - validBytes, 0U};
    DataCopyPad(dst, src, params);
}

__aicore__ inline void PreQuantCastBf16Weights(LocalTensor<uint8_t> stage, uint32_t tileCount, uint32_t topK,
                                               uint32_t weightSrcRowBytes, uint32_t weightDstRowBytes,
                                               uint32_t weightDstOffset)
{
    LocalTensor<bfloat16_t> src = stage.template ReinterpretCast<bfloat16_t>();
    LocalTensor<float> dst = stage[weightDstOffset].template ReinterpretCast<float>();
    // Cast 只写 topK 个有效值，先把每行的对齐尾部清零，随后可以按完整 FP32 行散写。
    LocalTensor<uint8_t> dstBytes = stage[weightDstOffset];
    Duplicate<uint8_t>(dstBytes, 0U, tileCount * weightDstRowBytes);
    PipeBarrier<PIPE_V>();
    for (uint32_t row = 0U; row < tileCount; ++row) {
        Cast(dst[row * (weightDstRowBytes / sizeof(float))],
             src[row * (weightSrcRowBytes / sizeof(bfloat16_t))], AscendC::RoundMode::CAST_NONE, topK);
    }
    PipeBarrier<PIPE_V>();
}

// 按 token -> scale -> topK weight 的顺序执行三个阶段。x 固定 B=1，另外两阶段使用 tiling 给出的合并数。
template <typename ActivationType, typename TopkWeightsType, bool TopkWeightsPrefetch>
__aicore__ inline void PackPreQuantizedTokenRange(const WorkRange &tokenRange, const MoeStageCommonConfig &common,
                                                  const Params &params, const QuantProcessConfig &config,
                                                  const PreQuantizedCopyContext &copyContext,
                                                  QuantProcessScratch<ActivationType> &scratch)
{
    // 穿刺版本固定 B=1；tiling 中保留该字段只是为了日志和后续扩展，不能让旧字段值改变
    // 当前 stage 的容量假设。
    constexpr uint32_t xTileTokens = 1U;
    const uint32_t scaleTileTokens = params.tilingData->preQuantScaleTileTokens == 0U
                                         ? 1U
                                         : params.tilingData->preQuantScaleTileTokens;
    const uint32_t weightTileTokens = params.tilingData->preQuantWeightTileTokens == 0U
                                          ? 1U
                                          : params.tilingData->preQuantWeightTileTokens;

    // x 阶段：每个 tile 只含一个 token，stage 直接对应一个通信 record 的 data 区。
    SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
    bool hasPending = false;
    uint32_t pendingStart = 0U;
    uint32_t pendingCount = 0U;
    TEventID pendingEvent = EVENT_ID0;
    for (uint32_t tileStart = 0U; tileStart < tokenRange.count; tileStart += xTileTokens) {
        uint32_t remaining = tokenRange.count - tileStart;
        uint32_t tileCount = xTileTokens < remaining ? xTileTokens : remaining;
        bool useFirstBuffer = ((tileStart / xTileTokens) % DOUBLE_BUFFER) == 0U;
        TEventID event = useFirstBuffer ? EVENT_ID0 : EVENT_ID1;
        LocalTensor<uint8_t> stage = useFirstBuffer ? scratch.preQuantStage0 : scratch.preQuantStage1;
        uint64_t dataOffset = static_cast<uint64_t>(tileStart) * copyContext.dataValidBytes;

        WaitFlag<AscendC::HardEvent::MTE3_MTE2>(event);
        PreQuantCopyIn(stage, copyContext.dataSrcGlobalTensor[dataOffset], tileCount, copyContext.dataValidBytes,
                       copyContext.dataRowBytes);
        SetFlag<AscendC::HardEvent::MTE2_MTE3>(event);

        if (hasPending) {
            LocalTensor<uint8_t> pendingStage =
                pendingEvent == EVENT_ID0 ? scratch.preQuantStage0 : scratch.preQuantStage1;
            uint64_t outputOffset = static_cast<uint64_t>(pendingStart) * config.quantTokenScaleAlignBytes;
            WaitFlag<AscendC::HardEvent::MTE2_MTE3>(pendingEvent);
            PreQuantScatter(copyContext.tokenScaleDstGlobalTensor[outputOffset], pendingStage, pendingCount,
                            copyContext.dataValidBytes,
                            Ops::Base::CeilAlign(copyContext.dataValidBytes, static_cast<uint32_t>(ALIGN_32)),
                            config.quantTokenScaleAlignBytes);
            SetFlag<AscendC::HardEvent::MTE3_MTE2>(pendingEvent);
        }
        hasPending = true;
        pendingStart = tileStart;
        pendingCount = tileCount;
        pendingEvent = event;
    }
    if (hasPending) {
        LocalTensor<uint8_t> pendingStage =
            pendingEvent == EVENT_ID0 ? scratch.preQuantStage0 : scratch.preQuantStage1;
        uint64_t outputOffset = static_cast<uint64_t>(pendingStart) * config.quantTokenScaleAlignBytes;
        WaitFlag<AscendC::HardEvent::MTE2_MTE3>(pendingEvent);
        PreQuantScatter(copyContext.tokenScaleDstGlobalTensor[outputOffset], pendingStage, pendingCount,
                        copyContext.dataValidBytes,
                        Ops::Base::CeilAlign(copyContext.dataValidBytes, static_cast<uint32_t>(ALIGN_32)),
                        config.quantTokenScaleAlignBytes);
        SetFlag<AscendC::HardEvent::MTE3_MTE2>(pendingEvent);
    }
    // scale 阶段：一个 token stage 可以容纳多个 scale row。
    hasPending = false;
    for (uint32_t tileStart = 0U; tileStart < tokenRange.count; tileStart += scaleTileTokens) {
        uint32_t remaining = tokenRange.count - tileStart;
        uint32_t tileCount = scaleTileTokens < remaining ? scaleTileTokens : remaining;
        bool useFirstBuffer = ((tileStart / scaleTileTokens) % DOUBLE_BUFFER) == 0U;
        TEventID event = useFirstBuffer ? EVENT_ID0 : EVENT_ID1;
        LocalTensor<uint8_t> stage = useFirstBuffer ? scratch.preQuantStage0 : scratch.preQuantStage1;
        uint64_t scaleOffset = static_cast<uint64_t>(tileStart) * copyContext.scaleValidBytes;

        WaitFlag<AscendC::HardEvent::MTE3_MTE2>(event);
        PreQuantCopyIn(stage, copyContext.scaleSrcGlobalTensor[scaleOffset], tileCount,
                       copyContext.scaleValidBytes, copyContext.scaleRowBytes);
        SetFlag<AscendC::HardEvent::MTE2_MTE3>(event);
        if (hasPending) {
            LocalTensor<uint8_t> pendingStage =
                pendingEvent == EVENT_ID0 ? scratch.preQuantStage0 : scratch.preQuantStage1;
            uint64_t outputOffset = static_cast<uint64_t>(pendingStart) * config.quantTokenScaleAlignBytes;
            WaitFlag<AscendC::HardEvent::MTE2_MTE3>(pendingEvent);
            PreQuantScatter(copyContext.tokenScaleDstGlobalTensor[outputOffset + config.quantTokenAlignBytes],
                            pendingStage, pendingCount, copyContext.scaleRowBytes, copyContext.scaleRowBytes,
                            config.quantTokenScaleAlignBytes);
            SetFlag<AscendC::HardEvent::MTE3_MTE2>(pendingEvent);
        }
        hasPending = true;
        pendingStart = tileStart;
        pendingCount = tileCount;
        pendingEvent = event;
    }
    if (hasPending) {
        LocalTensor<uint8_t> pendingStage =
            pendingEvent == EVENT_ID0 ? scratch.preQuantStage0 : scratch.preQuantStage1;
        uint64_t outputOffset = static_cast<uint64_t>(pendingStart) * config.quantTokenScaleAlignBytes;
        WaitFlag<AscendC::HardEvent::MTE2_MTE3>(pendingEvent);
        PreQuantScatter(copyContext.tokenScaleDstGlobalTensor[outputOffset + config.quantTokenAlignBytes], pendingStage,
                        pendingCount, copyContext.scaleRowBytes, copyContext.scaleRowBytes,
                        config.quantTokenScaleAlignBytes);
        SetFlag<AscendC::HardEvent::MTE3_MTE2>(pendingEvent);
    }
    WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);

    if constexpr (TopkWeightsPrefetch) {
        // topK weight 阶段最后执行；FP32 直接 MTE2->MTE3，BF16 先在 stage 内做 Cast。
        SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
        hasPending = false;
        uint32_t weightValidBytes = common.topK * sizeof(TopkWeightsType);
        uint32_t weightSrcRowBytes = Ops::Base::CeilAlign(weightValidBytes, static_cast<uint32_t>(ALIGN_32));
        uint32_t weightDstRowBytes = Ops::Base::CeilAlign(common.topK * static_cast<uint32_t>(sizeof(float)),
                                                          static_cast<uint32_t>(ALIGN_32));
        uint32_t weightOutputOffset = config.quantTokenAlignBytes + config.quantScaleAlignBytes;
        for (uint32_t tileStart = 0U; tileStart < tokenRange.count; tileStart += weightTileTokens) {
            uint32_t remaining = tokenRange.count - tileStart;
            uint32_t tileCount = weightTileTokens < remaining ? weightTileTokens : remaining;
            bool useFirstBuffer = ((tileStart / weightTileTokens) % DOUBLE_BUFFER) == 0U;
            TEventID event = useFirstBuffer ? EVENT_ID0 : EVENT_ID1;
            LocalTensor<uint8_t> stage = useFirstBuffer ? scratch.preQuantStage0 : scratch.preQuantStage1;
            uint64_t weightOffset =
                static_cast<uint64_t>(tokenRange.start + tileStart) * common.topK * sizeof(TopkWeightsType);
            GlobalTensor<uint8_t> weightGm;
            weightGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(params.probsGmAddr) + weightOffset);

            WaitFlag<AscendC::HardEvent::MTE3_MTE2>(event);
            PreQuantCopyIn(stage, weightGm, tileCount, weightValidBytes, weightSrcRowBytes);
            if constexpr (Std::IsSame<TopkWeightsType, bfloat16_t>::value) {
                SetFlag<AscendC::HardEvent::MTE2_V>(event);
            } else {
                SetFlag<AscendC::HardEvent::MTE2_MTE3>(event);
            }

            if (hasPending) {
                LocalTensor<uint8_t> pendingStage =
                    pendingEvent == EVENT_ID0 ? scratch.preQuantStage0 : scratch.preQuantStage1;
                uint64_t outputOffset = static_cast<uint64_t>(pendingStart) * config.quantTokenScaleAlignBytes;
                if constexpr (Std::IsSame<TopkWeightsType, bfloat16_t>::value) {
                    WaitFlag<AscendC::HardEvent::MTE2_V>(pendingEvent);
                    uint32_t castOffset = pendingCount * weightSrcRowBytes;
                    PreQuantCastBf16Weights(pendingStage, pendingCount, common.topK, weightSrcRowBytes,
                                            weightDstRowBytes, castOffset);
                    SetFlag<AscendC::HardEvent::V_MTE3>(pendingEvent);
                    WaitFlag<AscendC::HardEvent::V_MTE3>(pendingEvent);
                    PreQuantScatter(copyContext.tokenScaleDstGlobalTensor[outputOffset + weightOutputOffset],
                                    pendingStage[castOffset], pendingCount, weightDstRowBytes, weightDstRowBytes,
                                    config.quantTokenScaleAlignBytes);
                } else {
                    WaitFlag<AscendC::HardEvent::MTE2_MTE3>(pendingEvent);
                    PreQuantScatter(copyContext.tokenScaleDstGlobalTensor[outputOffset + weightOutputOffset],
                                    pendingStage, pendingCount, weightSrcRowBytes, weightSrcRowBytes,
                                    config.quantTokenScaleAlignBytes);
                }
                SetFlag<AscendC::HardEvent::MTE3_MTE2>(pendingEvent);
            }
            hasPending = true;
            pendingStart = tileStart;
            pendingCount = tileCount;
            pendingEvent = event;
        }
        if (hasPending) {
            LocalTensor<uint8_t> pendingStage =
                pendingEvent == EVENT_ID0 ? scratch.preQuantStage0 : scratch.preQuantStage1;
            uint64_t outputOffset = static_cast<uint64_t>(pendingStart) * config.quantTokenScaleAlignBytes;
            if constexpr (Std::IsSame<TopkWeightsType, bfloat16_t>::value) {
                WaitFlag<AscendC::HardEvent::MTE2_V>(pendingEvent);
                uint32_t castOffset = pendingCount * weightSrcRowBytes;
                PreQuantCastBf16Weights(pendingStage, pendingCount, common.topK, weightSrcRowBytes,
                                        weightDstRowBytes, castOffset);
                SetFlag<AscendC::HardEvent::V_MTE3>(pendingEvent);
                WaitFlag<AscendC::HardEvent::V_MTE3>(pendingEvent);
                PreQuantScatter(copyContext.tokenScaleDstGlobalTensor[outputOffset + weightOutputOffset],
                                pendingStage[castOffset], pendingCount, weightDstRowBytes, weightDstRowBytes,
                                config.quantTokenScaleAlignBytes);
            } else {
                WaitFlag<AscendC::HardEvent::MTE2_MTE3>(pendingEvent);
                PreQuantScatter(copyContext.tokenScaleDstGlobalTensor[outputOffset + weightOutputOffset], pendingStage,
                                pendingCount, weightSrcRowBytes, weightSrcRowBytes,
                                config.quantTokenScaleAlignBytes);
            }
            SetFlag<AscendC::HardEvent::MTE3_MTE2>(pendingEvent);
        }
        WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
    }
}

// 将调用方已量化的 x 和逐 32 元素 scale 按 dispatch 布局拼接，按需附加 topK 权重。
// 不对 x/scales 做数值转换。
template <typename XType, typename ActivationType, typename TopkWeightsType, bool TopkWeightsPrefetch>
__aicore__ inline void PackPreQuantizedLocalTokens(const AivJobContext &job, const MoeStageCommonConfig &common,
                                                   const Params &params, const QuantProcessConfig &config,
                                                   GM_ADDR outputAddr, QuantProcessScratch<ActivationType> &scratch)
{
    if constexpr (g_coreType == AIC) {
        return;
    }
    WorkRange tokenRange = TilingByJobContext(common.tokenNum, job.jobIndex, job.totalJobs, 1U);
    if (tokenRange.count == 0U) {
        return;
    }

    constexpr uint32_t X_ELEMS_PER_BYTE = PackedElementTraits<XType>::ELEMENTS_PER_BYTE;
    PreQuantizedCopyContext copyContext;
    copyContext.dataValidBytes = common.tokenHiddenDim / X_ELEMS_PER_BYTE;
    copyContext.scaleValidBytes = config.quantScaleValidCountPerToken * static_cast<uint32_t>(sizeof(fp8_e8m0_t));
    uint64_t dataRangeOffset =
        static_cast<uint64_t>(tokenRange.start) * static_cast<uint64_t>(copyContext.dataValidBytes);
    uint64_t scaleRangeOffset =
        static_cast<uint64_t>(tokenRange.start) * static_cast<uint64_t>(copyContext.scaleValidBytes);
    uint64_t tokenScaleRangeOffset =
        static_cast<uint64_t>(tokenRange.start) * static_cast<uint64_t>(config.quantTokenScaleAlignBytes);
    copyContext.dataSrcGlobalTensor.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(params.aGmAddr) +
                                                    dataRangeOffset);
    copyContext.scaleSrcGlobalTensor.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(params.xScaleGmAddr) +
                                                     scaleRangeOffset);
    copyContext.tokenScaleDstGlobalTensor.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(outputAddr) +
                                                          tokenScaleRangeOffset);

    // stage 行宽采用输出记录中对应字段的对齐宽度；x 的 scatter 只写有效字节，scale 则写完整对齐行。
    copyContext.dataRowBytes = config.quantTokenAlignBytes;
    copyContext.scaleRowBytes = config.quantScaleAlignBytes;
    PackPreQuantizedTokenRange<ActivationType, TopkWeightsType, TopkWeightsPrefetch>(tokenRange, common, params, config,
                                                                                     copyContext, scratch);
}

// 编译期根据 x 类型选择本卡 token/scale 数据准备方式：BF16 沿用动态量化，
// FP8/FP4 只打包调用方提供的 x/scales，不实例化输入量化分支。
template <typename XType, int32_t QuantMode, typename QuantOutType, typename ActivationType, typename TopkWeightsType,
          bool TopkWeightsPrefetch>
__aicore__ inline void PrepareLocalTokens(const AivJobContext &job, const MoeStageCommonConfig &common,
                                          const Params &params, const QuantProcessConfig &config, GM_ADDR outputAddr,
                                          QuantProcessScratch<ActivationType> &scratch)
{
    if constexpr (Std::IsSame<XType, bfloat16_t>::value) {
        QuantizeLocalTokens<QuantMode, QuantOutType, ActivationType, TopkWeightsType, TopkWeightsPrefetch>(
            job, common, params, config, outputAddr, scratch);
    } else {
        PackPreQuantizedLocalTokens<XType, ActivationType, TopkWeightsType, TopkWeightsPrefetch>(
            job, common, params, config, outputAddr, scratch);
    }
}

} // namespace MegaMoeImpl

#endif // MEGA_MOE_TOKEN_QUANT_H
