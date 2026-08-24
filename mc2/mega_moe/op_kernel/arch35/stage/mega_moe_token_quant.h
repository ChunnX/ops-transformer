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

#include "../common/mega_moe_mxfp8_utils.h"
#include "../common/mega_moe_utils.h"

namespace MegaMoeImpl {

using namespace AscendC;

struct QuantProcessConfig {
    uint32_t quantTokenAlignBytes;
    uint32_t quantScaleAlignBytes;
    uint32_t quantTokenScaleAlignBytes;
    uint32_t quantScaleNumAlignPerToken;
};

/*
 * 计算量化通信记录的对齐布局（普通与 wave 编排模板共用）。
 * 记录布局 = Align256(token 数据) + Align32(scale) + prefetch 时附加 Align32(topk 权重)，
 * 与 host CalcDispatchBufferConfig 的 copyBufferBytes 契约恒相等。
 */
template <typename ActivationType, typename QuantScaleOutType, bool TopkWeightsPrefetch, uint32_t AElemsPerByte>
__aicore__ inline QuantProcessConfig CreateQuantProcessConfig(uint32_t tokenHiddenDim, const Params &params)
{
    uint32_t quantScaleNumAlignPerToken = Ops::Base::CeilDiv(tokenHiddenDim, static_cast<uint32_t>(ALIGN_32));
    uint32_t quantTokenAlignBytes =
        Ops::Base::CeilAlign(tokenHiddenDim / AElemsPerByte, static_cast<uint32_t>(ALIGN_256)) * sizeof(ActivationType);
    uint32_t quantScaleAlignBytes = Ops::Base::CeilAlign(
        quantScaleNumAlignPerToken * static_cast<uint32_t>(sizeof(QuantScaleOutType)), static_cast<uint32_t>(ALIGN_32));
    uint32_t quantTokenScaleAlignBytes = quantTokenAlignBytes + quantScaleAlignBytes;
    if constexpr (TopkWeightsPrefetch) {
        uint32_t weightAlignBytes = Ops::Base::CeilAlign(static_cast<uint32_t>(params.tilingData->topK * sizeof(float)),
                                                         static_cast<uint32_t>(ALIGN_32));
        quantTokenScaleAlignBytes += weightAlignBytes;
    }
    return {quantTokenAlignBytes, quantScaleAlignBytes, quantTokenScaleAlignBytes, quantScaleNumAlignPerToken};
}

template <typename ActivationType>
struct QuantProcessScratch {
    LocalTensor<bfloat16_t> xInTensor0;
    LocalTensor<bfloat16_t> xInTensor1;
    LocalTensor<ActivationType> xOutTensor0;
    LocalTensor<ActivationType> xOutTensor1;
    LocalTensor<uint16_t> mxTempTensor;
};

template <typename TopkWeightsType, typename ActivationType, bool TopkWeightsPrefetch>
__aicore__ inline void LoadTopkWeightsToUb(const Params &params, const QuantProcessConfig &config,
                                           QuantProcessScratch<ActivationType> &scratch,
                                           const LocalTensor<ActivationType> &xOutTensor, int32_t tokenIndex,
                                           TEventID event)
{
    if constexpr (TopkWeightsPrefetch) {
        GlobalTensor<TopkWeightsType> weightGm;
        weightGm.SetGlobalBuffer(reinterpret_cast<__gm__ TopkWeightsType *>(
            params.probsGmAddr +
            static_cast<uint64_t>(tokenIndex) * params.tilingData->topK * sizeof(TopkWeightsType)));
        uint32_t weightOffsetInUb = config.quantTokenAlignBytes + config.quantScaleAlignBytes;
        if constexpr (Std::IsSame<TopkWeightsType, bfloat16_t>::value) {
            LocalTensor<TopkWeightsType> weightBf16Tmp =
                scratch.mxTempTensor.template ReinterpretCast<TopkWeightsType>();
            DataCopyPad(weightBf16Tmp, weightGm,
                        {1U, static_cast<uint32_t>(params.tilingData->topK * sizeof(TopkWeightsType)), 0U, 0U, 0U},
                        {false, 0U, 0U, 0U});
            SetFlag<AscendC::HardEvent::MTE2_V>(event);
            WaitFlag<AscendC::HardEvent::MTE2_V>(event);
            LocalTensor<float> weightFp32Ub = xOutTensor[weightOffsetInUb].template ReinterpretCast<float>();
            Cast(weightFp32Ub, weightBf16Tmp, AscendC::RoundMode::CAST_NONE, params.tilingData->topK);
            PipeBarrier<PIPE_V>();
        } else {
            LocalTensor<TopkWeightsType> weightUb =
                xOutTensor[weightOffsetInUb].template ReinterpretCast<TopkWeightsType>();
            DataCopyPad(weightUb, weightGm,
                        {1U, static_cast<uint32_t>(params.tilingData->topK * sizeof(TopkWeightsType)), 0U, 0U, 0U},
                        {false, 0U, 0U, 0U});
            SetFlag<AscendC::HardEvent::MTE2_V>(event);
            WaitFlag<AscendC::HardEvent::MTE2_V>(event);
        }
    }
}

// 原型：MegaMoe::QuantProcessInRank。量化一个逻辑 AIV 任务负责的本卡 token。
template <int32_t QuantMode, typename QuantOutType, typename ActivationType, typename TopkWeightsType,
          bool TopkWeightsPrefetch>
__aicore__ inline void QuantizeLocalTokens(const AivJobContext &job, const MoeStageCommonConfig &common,
                                           const Params &params, const QuantProcessConfig &config,
                                           QuantProcessScratch<ActivationType> &scratch)
{
    if constexpr (g_coreType == AIC) {
        return;
    }
    WorkRange tokenRange = TilingByJobContext(common.tokenNum, job.jobIndex, job.totalJobs, 1U);
    if (tokenRange.count == 0U) {
        return;
    }
    uint32_t hiddenDim = common.tokenHiddenDim;
    GlobalTensor<bfloat16_t> srcGlobalTensor;
    GlobalTensor<uint8_t> dstGlobalTensor;
    srcGlobalTensor.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(params.aGmAddr) +
                                    static_cast<uint64_t>(tokenRange.start) * hiddenDim);
    dstGlobalTensor.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(params.peermemInfo.quantTokenScalePtr) +
                                    static_cast<uint64_t>(tokenRange.start) * config.quantTokenScaleAlignBytes);
    DataCopyParams xCopyInParams = {1U, static_cast<uint16_t>(hiddenDim * sizeof(bfloat16_t)), 0U, 0U};
    DataCopyPadParams xCopyInPadParams{true, 0, 0, 0};
    DataCopyExtParams xCopyOutParams = {1U, config.quantTokenScaleAlignBytes, 0U, 0U, 0U};
    __ubuf__ uint16_t *maxExpAddr = reinterpret_cast<__ubuf__ uint16_t *>(scratch.mxTempTensor.GetPhyAddr());
    __ubuf__ uint16_t *halfScaleAddr = reinterpret_cast<__ubuf__ uint16_t *>(
        scratch.mxTempTensor[Ops::Base::CeilAlign(config.quantScaleNumAlignPerToken, static_cast<uint32_t>(ALIGN_32))]
            .GetPhyAddr());
    /*
     * h 非 64 元素对齐（h%64==32，即 scale 组数 scaleNum 为奇数）时，量化链路存在三处
     * "计算不覆盖、却会进入定长记录或参与计算"的跨 launch UB 残留：
     *  ① xIn 尾部（有效 h 元素之后到 CeilAlign(h,128) 末尾）：ComputeMaxExp 尾块仅 32 元素有效，
     *     但 mask 激活 lane0-31、经 deinterleave 覆盖元素位置 [h-32, h+32)，其中 [h, h+32) 为残留；
     *  ② xOut 记录内 scale 偶数补齐槽（第 scaleNum 个 scale 字节）：ComputeScale 的掩码写止于
     *     scale[scaleNum-1]，该槽永不被写入，随定长记录原样写出；
     *  ③ mxTempTensor 的 halfScale[scaleNum..] 补偶槽：同样永不被 ComputeScale 写入，却被
     *     ComputeFp8Data 尾块经 E2B 广播进 [h, h+32) 位置的乘法（0×Inf/NaN 仍为 NaN）。
     * GMM 侧按 64 元素 K 块 + e8m0 scale 成对（偶数个）消费该记录，上述残留一旦呈 NaN/大指数
     * （跨迭代复用 UB 时的上一轮残留即如此），整行 GMM 输出被污染为 NaN，最终 combine 输出成块清零。
     * 故此处把四个收发 buffer 与 mxTemp scratch 整体清零一次：有效区随后每 token 均被完整覆写，
     * 三处残留位从此恒为确定的 0（e8m0 0x00 = 2^-127、数据 0、halfScale 0，均为良性）；
     * h%64==0 时不存在上述缝隙，本清零不改变任何可观测行为。
     */
    auto xIn0View = scratch.xInTensor0.template ReinterpretCast<int16_t>();
    auto xIn1View = scratch.xInTensor1.template ReinterpretCast<int16_t>();
    auto xOut0View = scratch.xOutTensor0.template ReinterpretCast<int16_t>();
    auto xOut1View = scratch.xOutTensor1.template ReinterpretCast<int16_t>();
    auto mxTempView = scratch.mxTempTensor.template ReinterpretCast<int16_t>();
    Duplicate<int16_t>(xIn0View, 0, static_cast<int32_t>(xIn0View.GetSize()));
    Duplicate<int16_t>(xIn1View, 0, static_cast<int32_t>(xIn1View.GetSize()));
    Duplicate<int16_t>(xOut0View, 0, static_cast<int32_t>(xOut0View.GetSize()));
    Duplicate<int16_t>(xOut1View, 0, static_cast<int32_t>(xOut1View.GetSize()));
    Duplicate<int16_t>(mxTempView, 0, static_cast<int32_t>(mxTempView.GetSize()));
    PipeBarrier<PIPE_V>();
    SyncFuncStatic<AscendC::HardEvent::V_MTE2, SYNC_EVENT_ID2>();
    SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
    for (uint32_t index = 0; index < tokenRange.count; ++index) {
        bool useFirstBuffer = index % DOUBLE_BUFFER == 0;
        auto event = useFirstBuffer ? EVENT_ID0 : EVENT_ID1;
        auto xInTensor = useFirstBuffer ? scratch.xInTensor0 : scratch.xInTensor1;
        auto xOutTensor = useFirstBuffer ? scratch.xOutTensor0 : scratch.xOutTensor1;
        WaitFlag<AscendC::HardEvent::MTE3_MTE2>(event);
        DataCopyPad(xInTensor, srcGlobalTensor[static_cast<uint64_t>(index) * hiddenDim], xCopyInParams,
                    xCopyInPadParams);
        LoadTopkWeightsToUb<TopkWeightsType, ActivationType, TopkWeightsPrefetch>(params, config, scratch, xOutTensor,
                                                                                  tokenRange.start + index, event);
        if constexpr (!TopkWeightsPrefetch) {
            SetFlag<AscendC::HardEvent::MTE2_V>(event);
            WaitFlag<AscendC::HardEvent::MTE2_V>(event);
        }
        __ubuf__ bfloat16_t *srcAddr = (__ubuf__ bfloat16_t *)xInTensor.GetPhyAddr();
        __ubuf__ int8_t *outDataAddr = (__ubuf__ int8_t *)xOutTensor.GetPhyAddr();
        __ubuf__ uint16_t *mxScaleAddr = (__ubuf__ uint16_t *)xOutTensor[config.quantTokenAlignBytes].GetPhyAddr();

        if constexpr (QuantMode == E2M1_QUANT) {
            Quant::ComputeMaxExp(srcAddr, maxExpAddr, hiddenDim);
            Quant::ComputeScale<QuantOutType>(maxExpAddr, mxScaleAddr, halfScaleAddr,
                                              config.quantScaleNumAlignPerToken);
            Quant::ComputeFp4Data<bfloat16_t, QuantOutType, AscendC::RoundMode::CAST_TRUNC,
                                  AscendC::RoundMode::CAST_RINT>(srcAddr, halfScaleAddr, outDataAddr, hiddenDim);
        } else {
            Mxfp8::ComputeFp8Token<bfloat16_t, QuantOutType>(srcAddr, maxExpAddr, mxScaleAddr, halfScaleAddr,
                                                             outDataAddr, hiddenDim, config.quantScaleNumAlignPerToken);
        }
        SetFlag<AscendC::HardEvent::V_MTE3>(event);
        WaitFlag<AscendC::HardEvent::V_MTE3>(event);
        auto xOutBytesTensor = xOutTensor.template ReinterpretCast<uint8_t>();
        DataCopyPad(dstGlobalTensor[static_cast<uint64_t>(index) * config.quantTokenScaleAlignBytes], xOutBytesTensor,
                    xCopyOutParams);
        SetFlag<AscendC::HardEvent::MTE3_MTE2>(event);
    }
    WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
}

} // namespace MegaMoeImpl

#endif // MEGA_MOE_TOKEN_QUANT_H
