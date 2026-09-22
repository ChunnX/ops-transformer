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

// 仅由 AIV 调用，tokenRange 非空；将已量化的 x 和逐 32 元素 scale 按 dispatch 布局拼接，按需附加 topK 权重。
// 不对 x/scales 做数值转换。
template <typename XType, typename ActivationType, typename TopkWeightsType, bool TopkWeightsPrefetch>
__aicore__ inline void PackPreQuantizedLocalTokens(const WorkRange &tokenRange, const MoeStageCommonConfig &common,
                                                   const Params &params, const QuantProcessConfig &config,
                                                   GM_ADDR outputAddr, QuantProcessScratch<ActivationType> &scratch)
{
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

} // namespace MegaMoeImpl

#endif // MEGA_MOE_TOKEN_QUANT_H
