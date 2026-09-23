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
    GlobalTensor<bfloat16_t> inputGm;
    GlobalTensor<uint8_t> outputGm;
    LocalTensor<bfloat16_t> xInTensor0;
    LocalTensor<bfloat16_t> xInTensor1;
    LocalTensor<ActivationType> xOutTensor0;
    LocalTensor<ActivationType> xOutTensor1;
    LocalTensor<uint16_t> mxTempTensor;
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
        // 前两段保存 maxExp 和 halfScale，下一 token 的预取不能覆盖当前 token 的量化中间结果。
        // 权重暂存与输入使用同一双缓冲槽；H<=8192 时前两段最多 1024B，两个权重槽共 128B。
        uint32_t weightTempOffset =
            2U * Ops::Base::CeilAlign(config.quantScaleValidCountPerToken, static_cast<uint32_t>(ALIGN_32)) +
            static_cast<uint32_t>(event) * ALIGN_32;
        LocalTensor<TopkWeightsType> weightBf16Tmp =
            scratch.mxTempTensor[weightTempOffset].template ReinterpretCast<TopkWeightsType>();
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

// MoE 和共享专家使用相同的参数结构，量化类型在编译期确定。
template <int32_t Mode, typename OutType, typename StorageType>
struct TokenQuantParams {
    static constexpr int32_t QUANT_MODE = Mode;
    using QuantOutType = OutType;
    const QuantProcessConfig &config;
    QuantProcessScratch<StorageType> &scratch;
};

// maxExp 已由公共流程计算；这里只按目标格式计算 scale 和量化数据。
template <int32_t QuantMode, typename QuantOutType, typename ActivationType>
__aicore__ inline void QuantizeTokenInUb(const LocalTensor<bfloat16_t> &input,
                                         const LocalTensor<ActivationType> &xOutTensor,
                                         const LocalTensor<uint16_t> &mxTemp, const QuantProcessConfig &config,
                                         uint32_t hiddenDim)
{
    auto *maxExpAddr = reinterpret_cast<__ubuf__ uint16_t *>(mxTemp.GetPhyAddr());
    auto *halfScaleAddr = reinterpret_cast<__ubuf__ uint16_t *>(
        mxTemp[Ops::Base::CeilAlign(config.quantScaleValidCountPerToken, static_cast<uint32_t>(ALIGN_32))]
            .GetPhyAddr());
    auto *srcAddr = reinterpret_cast<__ubuf__ bfloat16_t *>(input.GetPhyAddr());
    auto *outDataAddr = reinterpret_cast<__ubuf__ int8_t *>(xOutTensor.GetPhyAddr());
    auto *mxScaleAddr = reinterpret_cast<__ubuf__ uint16_t *>(xOutTensor[config.quantTokenAlignBytes].GetPhyAddr());

    Quant::ComputeScale<QuantOutType>(maxExpAddr, mxScaleAddr, halfScaleAddr, config.quantScaleValidCountPerToken);
    if constexpr (QuantMode == E2M1_QUANT) {
        Quant::ComputeFp4Data<bfloat16_t, QuantOutType, AscendC::RoundMode::CAST_TRUNC, AscendC::RoundMode::CAST_RINT>(
            srcAddr, halfScaleAddr, outDataAddr, hiddenDim);
    } else {
        Quant::ComputeFp8Data<bfloat16_t, QuantOutType, AscendC::RoundMode::CAST_TRUNC, AscendC::RoundMode::CAST_RINT>(
            srcAddr, halfScaleAddr, outDataAddr, hiddenDim);
    }
}

// 仅由 AIV 调用，tokenRange 非空；量化指定范围的本卡 token，按逐 token 交织布局写入数据与 scale。
template <typename TopkWeightsType, bool TopkWeightsPrefetch, typename MoeQuantParams, typename... SharedQuantParams>
__aicore__ inline void QuantizeLocalTokens(const WorkRange &tokenRange, const MoeStageCommonConfig &common,
                                           GM_ADDR topkWeightsAddr, const MoeQuantParams &moeQuant,
                                           const SharedQuantParams &...sharedQuant)
{
    auto &scratch = moeQuant.scratch;
    const auto &config = moeQuant.config;
    uint32_t hiddenDim = common.tokenHiddenDim;
    // 量化 scratch（mxTemp/xOut0/xOut1/xIn0/xIn1）的跨 launch 残留清零由各编排的
    // SendAndQuantBuffInit 在分配处一次性完成（span 清零，与布局同源），见其注释。
    SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
    for (uint32_t index = 0; index < tokenRange.count; ++index) {
        bool useFirstBuffer = index % DOUBLE_BUFFER == 0;
        auto event = useFirstBuffer ? EVENT_ID0 : EVENT_ID1;
        auto xInTensor = useFirstBuffer ? scratch.xInTensor0 : scratch.xInTensor1;
        auto xOutTensor = useFirstBuffer ? scratch.xOutTensor0 : scratch.xOutTensor1;
        uint32_t tokenIndex = tokenRange.start + index;
        WaitFlag<AscendC::HardEvent::MTE3_MTE2>(event);
        DataCopyPad(xInTensor, scratch.inputGm[static_cast<uint64_t>(tokenIndex) * hiddenDim],
                    {1U, static_cast<uint16_t>(hiddenDim * sizeof(bfloat16_t)), 0U, 0U}, {true, 0, 0, 0});
        if constexpr (TopkWeightsPrefetch) {
            GM_ADDR tokenTopkWeightsAddr =
                topkWeightsAddr + static_cast<uint64_t>(tokenIndex) * common.topK * sizeof(TopkWeightsType);
            PrefetchTopkWeights<TopkWeightsType>(tokenTopkWeightsAddr, common.topK, config, scratch, xOutTensor, event);
        } else {
            SetFlag<AscendC::HardEvent::MTE2_V>(event);
            WaitFlag<AscendC::HardEvent::MTE2_V>(event);
        }
        // 同一 token 的输入与分组方式相同，最大指数只统计一次。
        Quant::ComputeMaxExp(reinterpret_cast<__ubuf__ bfloat16_t *>(xInTensor.GetPhyAddr()),
                             reinterpret_cast<__ubuf__ uint16_t *>(scratch.mxTempTensor.GetPhyAddr()), hiddenDim);
        // 每个目标使用相同的量化和搬出流程；空参数包不生成共享量化代码。
        auto processQuantOutput = [&](auto quant) __attribute__((cce_aicore))
        {
            using QuantParams = decltype(quant);
            const auto &outputConfig = quant.config;
            auto outputTensor = useFirstBuffer ? quant.scratch.xOutTensor0 : quant.scratch.xOutTensor1;
            QuantizeTokenInUb<QuantParams::QUANT_MODE, typename QuantParams::QuantOutType>(
                xInTensor, outputTensor, scratch.mxTempTensor, outputConfig, hiddenDim);
            SetFlag<HardEvent::V_MTE3>(event);
            WaitFlag<HardEvent::V_MTE3>(event);
            auto outputBytes = outputTensor.template ReinterpretCast<uint8_t>();
            DataCopyPad(
                quant.scratch.outputGm[static_cast<uint64_t>(tokenIndex) * outputConfig.quantTokenScaleAlignBytes],
                outputBytes, {1U, outputConfig.quantTokenScaleAlignBytes, 0U, 0U, 0U});
        };
        processQuantOutput(moeQuant);
        // sizeof... 统计参数包中的参数个数：不传共享量化参数时为 0，传入一份时为 1。
        // if constexpr 在编译期判断，无共享参数时不生成下面的代码。
        if constexpr (sizeof...(SharedQuantParams) > 0) {
            processQuantOutput(sharedQuant...);
        }
        // 有独立共享量化时，槽的释放排在共享输出写回之后。
        SetFlag<AscendC::HardEvent::MTE3_MTE2>(event);
    }
    WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
}

// 预量化搬入使用固定行宽和固定目标 stride；一个 DataCopyPad 搬完当前 tile 的一类数据。
__aicore__ inline void PreQuantCopyRows(LocalTensor<uint8_t> dst, GlobalTensor<uint8_t> src, uint32_t rowCount,
                                        uint32_t validBytes, uint32_t alignedBytes, uint32_t dstRowBytes)
{
    DataCopyExtParams copyParams{static_cast<uint16_t>(rowCount), validBytes, 0U, dstRowBytes - alignedBytes, 0U};
    DataCopyPadExtParams<uint8_t> padParams{true, 0U, 0U, 0U};
    DataCopyPad(dst, src, copyParams, padParams);
}

struct PreQuantizedCopyContext {
    uint32_t dataValidBytes;
    uint32_t dataAlignedBytes;
    uint32_t scaleValidBytes;
    uint32_t scaleAlignedBytes;
    uint32_t weightValidBytes;
    uint32_t weightSrcAlignedBytes;
    uint32_t weightAlignedBytes;
    uint32_t recordBytes;
    uint32_t weightOffsetBytes;
    GlobalTensor<uint8_t> dataSrcGlobalTensor;
    GlobalTensor<uint8_t> scaleSrcGlobalTensor;
    GlobalTensor<uint8_t> tokenScaleDstGlobalTensor;
};

template <typename TopkWeightsType, bool TopkWeightsPrefetch>
__aicore__ inline void PreQuantWriteTile(LocalTensor<uint8_t> stage, TEventID event, uint32_t tileCount,
                                          uint32_t topK, uint32_t recordBytes, uint32_t weightSrcOffset,
                                          uint32_t weightSrcRowBytes, uint32_t weightOffsetBytes,
                                          GlobalTensor<uint8_t> output)
{
    if constexpr (TopkWeightsPrefetch && Std::IsSame<TopkWeightsType, bfloat16_t>::value) {
        WaitFlag<AscendC::HardEvent::MTE2_V>(event);
        LocalTensor<bfloat16_t> weightSrc = stage[weightSrcOffset].template ReinterpretCast<bfloat16_t>();
        for (uint32_t row = 0U; row < tileCount; ++row) {
            LocalTensor<float> weightDst =
                stage[row * recordBytes + weightOffsetBytes].template ReinterpretCast<float>();
            Cast(weightDst, weightSrc[row * (weightSrcRowBytes / sizeof(bfloat16_t))],
                 AscendC::RoundMode::CAST_NONE, topK);
        }
        PipeBarrier<PIPE_V>();
        SetFlag<AscendC::HardEvent::V_MTE3>(event);
        WaitFlag<AscendC::HardEvent::V_MTE3>(event);
    } else {
        WaitFlag<AscendC::HardEvent::MTE2_MTE3>(event);
    }

    DataCopyPad(output, stage,
                {static_cast<uint16_t>(tileCount), recordBytes, 0U, 0U, 0U});
    SetFlag<AscendC::HardEvent::MTE3_MTE2>(event);
}

// 每个 tile 搬入 B 个 token/scale/weight，完成后一次 MTE3 写出 B 条完整记录。
template <typename ActivationType, typename TopkWeightsType, bool TopkWeightsPrefetch>
__aicore__ inline void PackPreQuantizedTokenRange(const WorkRange &tokenRange, const MoeStageCommonConfig &common,
                                                  const Params &params, const PreQuantizedCopyContext &copyContext,
                                                  QuantProcessScratch<ActivationType> &scratch)
{
    const uint32_t tileCapacity = params.tilingData->preQuantTileTokens;
    const uint32_t recordBytes = copyContext.recordBytes;
    const uint32_t weightSrcOffset = tileCapacity * recordBytes;

    SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
    bool hasPendingTile = false;
    uint32_t pendingTileStart = 0U;
    uint32_t pendingTileCount = 0U;
    TEventID pendingEvent = EVENT_ID0;

    for (uint32_t tileStart = 0U; tileStart < tokenRange.count; tileStart += tileCapacity) {
        const uint32_t tileOrdinal = tileStart / tileCapacity;
        const uint32_t remainingTokens = tokenRange.count - tileStart;
        const uint32_t tileCount = remainingTokens < tileCapacity ? remainingTokens : tileCapacity;
        const TEventID event = (tileOrdinal & 1U) == 0U ? EVENT_ID0 : EVENT_ID1;
        LocalTensor<ActivationType> stageTensor = event == EVENT_ID0 ? scratch.xOutTensor0 : scratch.xOutTensor1;
        LocalTensor<uint8_t> stage = stageTensor.template ReinterpretCast<uint8_t>();

        WaitFlag<AscendC::HardEvent::MTE3_MTE2>(event);
        const uint64_t dataOffset = static_cast<uint64_t>(tileStart) * copyContext.dataValidBytes;
        const uint64_t scaleOffset = static_cast<uint64_t>(tileStart) * copyContext.scaleValidBytes;
        PreQuantCopyRows(stage, copyContext.dataSrcGlobalTensor[dataOffset], tileCount, copyContext.dataValidBytes,
                         copyContext.dataAlignedBytes, recordBytes);
        PreQuantCopyRows(stage[params.tilingData->preQuantTokenAlignedBytes],
                         copyContext.scaleSrcGlobalTensor[scaleOffset], tileCount, copyContext.scaleValidBytes,
                         copyContext.scaleAlignedBytes, recordBytes);

        if constexpr (TopkWeightsPrefetch) {
            const uint32_t tokenIndex = tokenRange.start + tileStart;
            const uint64_t weightOffset =
                static_cast<uint64_t>(tokenIndex) * static_cast<uint64_t>(copyContext.weightValidBytes);
            GlobalTensor<uint8_t> weightGm;
            weightGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(params.probsGmAddr) + weightOffset);
            if constexpr (Std::IsSame<TopkWeightsType, bfloat16_t>::value) {
                PreQuantCopyRows(stage[weightSrcOffset], weightGm, tileCount, copyContext.weightValidBytes,
                                 copyContext.weightSrcAlignedBytes, copyContext.weightSrcAlignedBytes);
                SetFlag<AscendC::HardEvent::MTE2_V>(event);
            } else {
                PreQuantCopyRows(stage[copyContext.weightOffsetBytes], weightGm, tileCount,
                                 copyContext.weightValidBytes, copyContext.weightAlignedBytes, recordBytes);
                SetFlag<AscendC::HardEvent::MTE2_MTE3>(event);
            }
        } else {
            SetFlag<AscendC::HardEvent::MTE2_MTE3>(event);
        }

        if (hasPendingTile) {
            LocalTensor<ActivationType> pendingStageTensor =
                pendingEvent == EVENT_ID0 ? scratch.xOutTensor0 : scratch.xOutTensor1;
            LocalTensor<uint8_t> pendingStage = pendingStageTensor.template ReinterpretCast<uint8_t>();
            const uint64_t outputOffset = static_cast<uint64_t>(pendingTileStart) * recordBytes;
            PreQuantWriteTile<TopkWeightsType, TopkWeightsPrefetch>(
                pendingStage, pendingEvent, pendingTileCount, common.topK, recordBytes, weightSrcOffset,
                copyContext.weightSrcAlignedBytes, copyContext.weightOffsetBytes,
                copyContext.tokenScaleDstGlobalTensor[outputOffset]);
        }

        hasPendingTile = true;
        pendingTileStart = tileStart;
        pendingTileCount = tileCount;
        pendingEvent = event;
    }

    if (hasPendingTile) {
        LocalTensor<ActivationType> pendingStageTensor =
            pendingEvent == EVENT_ID0 ? scratch.xOutTensor0 : scratch.xOutTensor1;
        LocalTensor<uint8_t> pendingStage = pendingStageTensor.template ReinterpretCast<uint8_t>();
        const uint64_t outputOffset = static_cast<uint64_t>(pendingTileStart) * recordBytes;
        PreQuantWriteTile<TopkWeightsType, TopkWeightsPrefetch>(
            pendingStage, pendingEvent, pendingTileCount, common.topK, recordBytes, weightSrcOffset,
            copyContext.weightSrcAlignedBytes, copyContext.weightOffsetBytes,
            copyContext.tokenScaleDstGlobalTensor[outputOffset]);
    }
    WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
}

// 仅由 AIV 调用，布局参数由 tilingData 提供；动态量化路径保持独立。
template <typename ActivationType, typename TopkWeightsType, bool TopkWeightsPrefetch>
__aicore__ inline void PackPreQuantizedLocalTokens(const WorkRange &tokenRange, const MoeStageCommonConfig &common,
                                                   const Params &params, GM_ADDR outputAddr,
                                                   QuantProcessScratch<ActivationType> &scratch)
{
    const auto *tilingData = params.tilingData;
    PreQuantizedCopyContext copyContext;
    copyContext.dataValidBytes = tilingData->preQuantTokenValidBytes;
    copyContext.dataAlignedBytes = tilingData->preQuantTokenAlignedBytes;
    copyContext.scaleValidBytes = tilingData->preQuantScaleValidBytes;
    copyContext.scaleAlignedBytes = tilingData->preQuantScaleAlignedBytes;
    copyContext.weightValidBytes = tilingData->preQuantWeightSrcBytes;
    copyContext.weightSrcAlignedBytes = tilingData->preQuantWeightSrcAlignedBytes;
    copyContext.weightAlignedBytes = tilingData->preQuantWeightAlignedBytes;
    copyContext.recordBytes = tilingData->preQuantRecordBytes;
    copyContext.weightOffsetBytes = tilingData->preQuantWeightOffsetBytes;

    const uint64_t dataRangeOffset = static_cast<uint64_t>(tokenRange.start) * copyContext.dataValidBytes;
    const uint64_t scaleRangeOffset = static_cast<uint64_t>(tokenRange.start) * copyContext.scaleValidBytes;
    const uint64_t outputRangeOffset = static_cast<uint64_t>(tokenRange.start) * copyContext.recordBytes;
    copyContext.dataSrcGlobalTensor.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(params.aGmAddr) +
                                                    dataRangeOffset);
    copyContext.scaleSrcGlobalTensor.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(params.xScaleGmAddr) +
                                                     scaleRangeOffset);
    copyContext.tokenScaleDstGlobalTensor.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(outputAddr) +
                                                          outputRangeOffset);

    PackPreQuantizedTokenRange<ActivationType, TopkWeightsType, TopkWeightsPrefetch>(tokenRange, common, params,
                                                                                     copyContext, scratch);
}

} // namespace MegaMoeImpl

#endif // MEGA_MOE_TOKEN_QUANT_H
