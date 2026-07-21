/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MEGA_MOE_QUANT_PROCESS_H
#define MEGA_MOE_QUANT_PROCESS_H

#include "../single_core_context.h"
#if __has_include("../../../moe_distribute_dispatch_v2/quantize_functions.h")
#include "../../../moe_distribute_dispatch_v2/quantize_functions.h"
#else
#include "../../../../moe_distribute_dispatch_v2/op_kernel/quantize_functions.h"
#endif

namespace MegaMoeImpl {

struct QuantProcessArgs {
    GM_ADDR xGmAddr;
    GM_ADDR quantTokenScalePtr;
    uint32_t quantTokenAlignBytes;
    uint32_t quantScaleAlignBytes;
    uint32_t quantTokenScaleAlignBytes;
    uint32_t quantScaleNumAlignPerToken;
};

template <typename ActivationType>
struct QuantProcessScratch {
    LocalTensor<bfloat16_t> xInTensor0;
    LocalTensor<bfloat16_t> xInTensor1;
    LocalTensor<ActivationType> xOutTensor0;
    LocalTensor<ActivationType> xOutTensor1;
    LocalTensor<uint16_t> mxTempTensor;
};

// Prototype: MegaMoe::QuantProcessInRank. Quantizes the local tokens assigned to one logical AIV thread.
template <int32_t QuantMode, typename QuantOutType, typename ActivationType>
__aicore__ inline void QuantProcessInRankOneCore(const DispatchPrepareContext &context,
    const QuantProcessArgs &args, QuantProcessScratch<ActivationType> &scratch)
{
    if constexpr(g_coreType == AIC) {
        return;
    }
    const AivThreadContext &thread = context.thread;
    int32_t currentNum;
    int32_t currentOffset;
    TilingByThreadContext(
        context.tokenShape.tokenNum, currentNum, currentOffset, thread.jobIndex, thread.totalJobs, 1);
    if (currentNum <= 0) {
        return;
    }
    const uint32_t hiddenDim = context.tokenShape.tokenHiddenDim;
    const uint32_t quantTokenAlignBytes = args.quantTokenAlignBytes;
    const uint32_t quantScaleAlignBytes = args.quantScaleAlignBytes;
    const uint32_t quantTokenScaleAlignBytes = args.quantTokenScaleAlignBytes;
    const uint32_t quantScaleNumAlignPerToken = args.quantScaleNumAlignPerToken;
    GlobalTensor<bfloat16_t> srcGlobalTensor;
    GlobalTensor<uint8_t> dstGlobalTensor;
    srcGlobalTensor.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(args.xGmAddr) +
        static_cast<uint64_t>(currentOffset) * hiddenDim);
    dstGlobalTensor.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(args.quantTokenScalePtr) +
        static_cast<uint64_t>(currentOffset) * quantTokenScaleAlignBytes);
    DataCopyParams xCopyInParams = {1U, static_cast<uint16_t>(hiddenDim * sizeof(bfloat16_t)), 0U, 0U};
    DataCopyPadParams xCopyInPadParams{true, 0, 0, 0};
    DataCopyExtParams xCopyOutParams = {1U,
        quantTokenAlignBytes + quantScaleAlignBytes, 0U, 0U, 0U};
    __ubuf__ uint16_t* maxExpAddr = reinterpret_cast<__ubuf__ uint16_t *>(scratch.mxTempTensor.GetPhyAddr());
    __ubuf__ uint16_t* halfScaleAddr = reinterpret_cast<__ubuf__ uint16_t *>(
        scratch.mxTempTensor[Ops::Base::CeilAlign(
            quantScaleNumAlignPerToken, static_cast<uint32_t>(ALIGN_32))].GetPhyAddr());
    SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
    for (int32_t index = 0; index < currentNum; index++) {
        bool useFirstBuffer = index % DOUBLE_BUFFER == 0;
        auto event = useFirstBuffer ? EVENT_ID0 : EVENT_ID1;
        auto xInTensor = useFirstBuffer ? scratch.xInTensor0 : scratch.xInTensor1;
        auto xOutTensor = useFirstBuffer ? scratch.xOutTensor0 : scratch.xOutTensor1;
        WaitFlag<AscendC::HardEvent::MTE3_MTE2>(event);
        DataCopyPad(xInTensor, srcGlobalTensor[static_cast<uint64_t>(index) * hiddenDim],
            xCopyInParams, xCopyInPadParams);
        SetFlag<AscendC::HardEvent::MTE2_V>(event);
        WaitFlag<AscendC::HardEvent::MTE2_V>(event);
        __ubuf__ bfloat16_t* srcAddr = (__ubuf__ bfloat16_t*)xInTensor.GetPhyAddr();
        __ubuf__ int8_t* outDataAddr = (__ubuf__ int8_t*)xOutTensor.GetPhyAddr();
        __ubuf__ uint16_t* mxScaleAddr = (__ubuf__ uint16_t*)xOutTensor[quantTokenAlignBytes].GetPhyAddr();

        Quant::ComputeMaxExp(srcAddr, maxExpAddr, hiddenDim);
        Quant::ComputeScale<QuantOutType>(maxExpAddr, mxScaleAddr, halfScaleAddr,
            quantScaleNumAlignPerToken);
        if constexpr (QuantMode == E2M1_QUANT) {
            Quant::ComputeFp4Data<bfloat16_t, QuantOutType, AscendC::RoundMode::CAST_TRUNC,
                AscendC::RoundMode::CAST_RINT>(srcAddr, halfScaleAddr, outDataAddr, hiddenDim);
        } else {
            Quant::ComputeFp8Data<bfloat16_t, QuantOutType, AscendC::RoundMode::CAST_TRUNC,
                AscendC::RoundMode::CAST_RINT>(srcAddr, halfScaleAddr, outDataAddr, hiddenDim);
        }
        SetFlag<AscendC::HardEvent::V_MTE3>(event);
        WaitFlag<AscendC::HardEvent::V_MTE3>(event);
        auto xOutBytesTensor = xOutTensor.template ReinterpretCast<uint8_t>();
        DataCopyPad(dstGlobalTensor[static_cast<uint64_t>(index) * quantTokenScaleAlignBytes],
            xOutBytesTensor, xCopyOutParams);
        SetFlag<AscendC::HardEvent::MTE3_MTE2>(event);
    }
    WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
}

} // namespace MegaMoeImpl

#endif // MEGA_MOE_QUANT_PROCESS_H
