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

// B=1 预量化搬运只保留两个最大字段 slot，token、scale、weight 分阶段复用。
struct PreQuantizedB1Context {
    uint32_t tokenValidBytes;
    uint32_t tokenAlignedBytes;
    uint32_t scaleValidBytes;
    uint32_t scaleAlignedBytes;
    uint32_t weightSrcBytes;
    uint32_t weightSrcAlignedBytes;
    uint32_t weightAlignedBytes;
    uint32_t tokenSrcStrideBytes;
    uint32_t scaleSrcStrideBytes;
    uint32_t weightSrcStrideBytes;
    uint32_t tokenOffsetBytes;
    uint32_t scaleOffsetBytes;
    uint32_t weightOffsetBytes;
    uint32_t recordBytes;
    GlobalTensor<uint8_t> dataSrcGlobalTensor;
    GlobalTensor<uint8_t> scaleSrcGlobalTensor;
    GlobalTensor<uint8_t> weightSrcGlobalTensor;
    GlobalTensor<uint8_t> outputGlobalTensor;
};

__aicore__ inline void PreQuantCopyField(LocalTensor<uint8_t> slot, GlobalTensor<uint8_t> src, uint64_t srcOffset,
                                         uint32_t validBytes, uint32_t alignedBytes, TEventID event,
                                         GlobalTensor<uint8_t> output, uint64_t outputOffset)
{
    WaitFlag<AscendC::HardEvent::MTE3_MTE2>(event);
    DataCopyPad(slot, src[srcOffset], {1U, validBytes, 0U, 0U, 0U}, {true, 0U, 0U, 0U});
    SetFlag<AscendC::HardEvent::MTE2_MTE3>(event);
    WaitFlag<AscendC::HardEvent::MTE2_MTE3>(event);
    DataCopyPad(output[outputOffset], slot, {1U, alignedBytes, 0U, 0U, 0U});
    SetFlag<AscendC::HardEvent::MTE3_MTE2>(event);
}

template <typename ActivationType>
__aicore__ inline void PreQuantCopyPhase(uint32_t tokenCount, GlobalTensor<uint8_t> src, uint32_t srcStrideBytes,
                                         uint32_t validBytes, uint32_t alignedBytes, uint32_t outputOffsetBytes,
                                         uint32_t recordBytes, GlobalTensor<uint8_t> output,
                                         QuantProcessScratch<ActivationType> &scratch)
{
    for (uint32_t index = 0U; index < tokenCount; ++index) {
        const TEventID event = (index & 1U) == 0U ? EVENT_ID0 : EVENT_ID1;
        LocalTensor<ActivationType> slotTensor = event == EVENT_ID0 ? scratch.xOutTensor0 : scratch.xOutTensor1;
        LocalTensor<uint8_t> slot = slotTensor.template ReinterpretCast<uint8_t>();
        const uint64_t sourceOffset = static_cast<uint64_t>(index) * srcStrideBytes;
        const uint64_t outputOffset = static_cast<uint64_t>(index) * recordBytes + outputOffsetBytes;
        PreQuantCopyField(slot, src, sourceOffset, validBytes, alignedBytes, event, output, outputOffset);
    }
}

template <typename ActivationType, typename TopkWeightsType, bool TopkWeightsPrefetch>
__aicore__ inline void PackPreQuantizedTokenRange(const WorkRange &tokenRange, const MoeStageCommonConfig &common,
                                                  const PreQuantizedB1Context &copyContext,
                                                  QuantProcessScratch<ActivationType> &scratch)
{
    SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);

    // 阶段一：逐 token 搬入并写回 token 字段。
    PreQuantCopyPhase<ActivationType>(tokenRange.count, copyContext.dataSrcGlobalTensor,
                                      copyContext.tokenSrcStrideBytes, copyContext.tokenValidBytes,
                                      copyContext.tokenAlignedBytes, copyContext.tokenOffsetBytes,
                                      copyContext.recordBytes, copyContext.outputGlobalTensor, scratch);

    // 阶段二：逐 token 搬入并写回 scale 字段。
    PreQuantCopyPhase<ActivationType>(tokenRange.count, copyContext.scaleSrcGlobalTensor,
                                      copyContext.scaleSrcStrideBytes, copyContext.scaleValidBytes,
                                      copyContext.scaleAlignedBytes, copyContext.scaleOffsetBytes,
                                      copyContext.recordBytes, copyContext.outputGlobalTensor, scratch);

    if constexpr (TopkWeightsPrefetch) {
        if constexpr (Std::IsSame<TopkWeightsType, bfloat16_t>::value) {
            // BF16 weight 先放入 mxTemp，再转换到 slot，避免输入和 float 输出重叠。
            for (uint32_t index = 0U; index < tokenRange.count; ++index) {
                const TEventID event = (index & 1U) == 0U ? EVENT_ID0 : EVENT_ID1;
                LocalTensor<ActivationType> slotTensor = event == EVENT_ID0 ? scratch.xOutTensor0 : scratch.xOutTensor1;
                LocalTensor<uint8_t> slot = slotTensor.template ReinterpretCast<uint8_t>();
                const uint32_t tempOffset = static_cast<uint32_t>(event) *
                                             (copyContext.weightSrcAlignedBytes / sizeof(uint16_t));
                LocalTensor<bfloat16_t> weightTemp =
                    scratch.mxTempTensor[tempOffset].template ReinterpretCast<bfloat16_t>();
                const uint64_t sourceOffset = static_cast<uint64_t>(index) * copyContext.weightSrcStrideBytes;
                const uint64_t outputOffset = static_cast<uint64_t>(index) * copyContext.recordBytes +
                                              copyContext.weightOffsetBytes;
                WaitFlag<AscendC::HardEvent::MTE3_MTE2>(event);
                DataCopyPad(weightTemp, copyContext.weightSrcGlobalTensor[sourceOffset],
                            {1U, copyContext.weightSrcBytes, 0U, 0U, 0U}, {true, 0U, 0U, 0U});
                SetFlag<AscendC::HardEvent::MTE2_V>(event);
                WaitFlag<AscendC::HardEvent::MTE2_V>(event);
                Duplicate<uint8_t>(slot, 0U, copyContext.weightAlignedBytes);
                LocalTensor<float> weightOutput = slot.template ReinterpretCast<float>();
                Cast(weightOutput, weightTemp, AscendC::RoundMode::CAST_NONE, common.topK);
                PipeBarrier<PIPE_V>();
                SetFlag<AscendC::HardEvent::V_MTE3>(event);
                WaitFlag<AscendC::HardEvent::V_MTE3>(event);
                DataCopyPad(copyContext.outputGlobalTensor[outputOffset], slot,
                            {1U, copyContext.weightAlignedBytes, 0U, 0U, 0U});
                SetFlag<AscendC::HardEvent::MTE3_MTE2>(event);
            }
        } else {
            // FP32 weight 不需要转换，直接复用普通字段搬运。
            PreQuantCopyPhase<ActivationType>(tokenRange.count, copyContext.weightSrcGlobalTensor,
                                               copyContext.weightSrcStrideBytes, copyContext.weightSrcBytes,
                                               copyContext.weightAlignedBytes, copyContext.weightOffsetBytes,
                                               copyContext.recordBytes, copyContext.outputGlobalTensor, scratch);
        }
    }

    WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
}

// 仅由 AIV 调用；布局长度、stride、offset 和 slot 大小全部由 tilingData 提供。
template <typename ActivationType, typename TopkWeightsType, bool TopkWeightsPrefetch>
__aicore__ inline void PackPreQuantizedLocalTokens(const WorkRange &tokenRange, const MoeStageCommonConfig &common,
                                                   const Params &params, GM_ADDR outputAddr,
                                                   QuantProcessScratch<ActivationType> &scratch)
{
    const auto *tilingData = params.tilingData;
    PreQuantizedB1Context copyContext;
    copyContext.tokenValidBytes = tilingData->preQuantTokenValidBytes;
    copyContext.tokenAlignedBytes = tilingData->preQuantTokenAlignedBytes;
    copyContext.scaleValidBytes = tilingData->preQuantScaleValidBytes;
    copyContext.scaleAlignedBytes = tilingData->preQuantScaleAlignedBytes;
    copyContext.weightSrcBytes = tilingData->preQuantWeightSrcBytes;
    copyContext.weightSrcAlignedBytes = tilingData->preQuantWeightSrcAlignedBytes;
    copyContext.weightAlignedBytes = tilingData->preQuantWeightAlignedBytes;
    copyContext.tokenSrcStrideBytes = tilingData->preQuantTokenSrcStrideBytes;
    copyContext.scaleSrcStrideBytes = tilingData->preQuantScaleSrcStrideBytes;
    copyContext.weightSrcStrideBytes = tilingData->preQuantWeightSrcStrideBytes;
    copyContext.tokenOffsetBytes = tilingData->preQuantTokenOffsetBytes;
    copyContext.scaleOffsetBytes = tilingData->preQuantScaleOffsetBytes;
    copyContext.weightOffsetBytes = tilingData->preQuantWeightOffsetBytes;
    copyContext.recordBytes = tilingData->preQuantRecordBytes;

    const uint64_t tokenRangeDataOffset =
        static_cast<uint64_t>(tokenRange.start) * copyContext.tokenSrcStrideBytes;
    const uint64_t tokenRangeScaleOffset =
        static_cast<uint64_t>(tokenRange.start) * copyContext.scaleSrcStrideBytes;
    const uint64_t tokenRangeWeightOffset =
        static_cast<uint64_t>(tokenRange.start) * copyContext.weightSrcStrideBytes;
    const uint64_t tokenRangeOutputOffset =
        static_cast<uint64_t>(tokenRange.start) * copyContext.recordBytes;
    copyContext.dataSrcGlobalTensor.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(params.aGmAddr) +
                                                    tokenRangeDataOffset);
    copyContext.scaleSrcGlobalTensor.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(params.xScaleGmAddr) +
                                                     tokenRangeScaleOffset);
    copyContext.weightSrcGlobalTensor.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(params.probsGmAddr) +
                                                      tokenRangeWeightOffset);
    copyContext.outputGlobalTensor.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(outputAddr) +
                                                   tokenRangeOutputOffset);
    PackPreQuantizedTokenRange<ActivationType, TopkWeightsType, TopkWeightsPrefetch>(tokenRange, common, copyContext,
                                                                                     scratch);
}

} // namespace MegaMoeImpl

#endif // MEGA_MOE_TOKEN_QUANT_H
