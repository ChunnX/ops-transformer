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
 * \file li_v2_vector_service_common.h
 * \brief LightningIndexerV2 与 QuantLightningIndexerV2 共用的 vector 服务函数
 *        Shared vector service functions used by LightningIndexerV2 and QuantLightningIndexerV2
 */
#ifndef LIGHTNING_INDEXER_V2_VECTOR_SERVICE_COMMON_H
#define LIGHTNING_INDEXER_V2_VECTOR_SERVICE_COMMON_H

#include "kernel_operator.h"
#include "li_v2_common_utils.h"

using namespace AscendC;

namespace liVectorServiceCommon {

using namespace LICommon;

// 有效序列长度为0时，将输出刷新为 INVALID_IDX / NEG_INF
// Flush output to INVALID_IDX / NEG_INF when the actual sequence length is zero
template <typename VALUE_T, bool LEADING_MTE3_SET = false, bool TRAILING_MTE3_SET = true>
__aicore__ inline void CleanInvalidOutput(GlobalTensor<int32_t> &indiceOutGm, const GlobalTensor<VALUE_T> &valueOutGm,
                                          LocalTensor<VALUE_T> &valueOutLocal, int64_t invalidS1Offset,
                                          uint64_t topkCount, int32_t invalidIdx, uint64_t negInfBits,
                                          bool returnValueFlagEnabled, uint32_t topkMte3VEvent, uint32_t topkVMTE3Event)
{
    using NegInfTypeT = std::conditional_t<std::is_same_v<VALUE_T, float>, uint32_t, uint16_t>;
    // init -1 and copy to output
    uint64_t dealSize = topkCount;
    GlobalTensor<int32_t> indexOutput = indiceOutGm[invalidS1Offset];
    AscendC::InitGlobalMemory(indexOutput, dealSize, invalidIdx);

    if (returnValueFlagEnabled) {
        if constexpr (LEADING_MTE3_SET) {
            SetFlag<HardEvent::MTE3_V>(topkMte3VEvent);
        }
        WaitFlag<HardEvent::MTE3_V>(topkMte3VEvent);
        Duplicate(valueOutLocal.template ReinterpretCast<NegInfTypeT>(), (NegInfTypeT)(negInfBits), topkCount);

        SetFlag<HardEvent::V_MTE3>(topkVMTE3Event);
        WaitFlag<HardEvent::V_MTE3>(topkVMTE3Event);

        AscendC::DataCopyParams copyOutValueParams;
        copyOutValueParams.blockCount = 1;
        copyOutValueParams.blockLen = topkCount * sizeof(VALUE_T);
        copyOutValueParams.srcStride = 0;
        copyOutValueParams.dstStride = 0;
        AscendC::DataCopyPad(valueOutGm[invalidS1Offset], valueOutLocal, copyOutValueParams);
        if constexpr (TRAILING_MTE3_SET) {
            SetFlag<HardEvent::MTE3_V>(topkMte3VEvent);
        }
    }
}

// 输出做 TND padding：索引区刷 INVALID_IDX、value 区刷 NEG_INF
// TND padding on the output: index flushed to INVALID_IDX, value flushed to NEG_INF
template <typename VALUE_T>
__aicore__ inline void DoTndPadding(GlobalTensor<int32_t> &indiceOutGm, const GlobalTensor<VALUE_T> &valueOutGm,
                                    uint32_t curCuSeqlensQ, uint32_t curSequsedQ, uint64_t indiceOutOffset,
                                    uint64_t kHeadNum, uint64_t topkCount, bool returnValueFlagEnabled,
                                    int32_t invalidIdx, uint64_t negInfBits, uint32_t topkMte3VEvent)
{
    using NegInfTypeT = std::conditional_t<std::is_same_v<VALUE_T, float>, uint32_t, uint16_t>;
    uint32_t paddingLen = curCuSeqlensQ - curSequsedQ;
    uint64_t paddingOffset = indiceOutOffset + curSequsedQ * kHeadNum * topkCount;
    uint64_t dealSize = paddingLen * kHeadNum * topkCount;
    GlobalTensor<int32_t> indiceOutPaddingStart = indiceOutGm[paddingOffset];
    AscendC::InitGlobalMemory(indiceOutPaddingStart, dealSize, invalidIdx);

    if (returnValueFlagEnabled) {
        SetFlag<HardEvent::MTE3_V>(topkMte3VEvent);
        WaitFlag<HardEvent::MTE3_V>(topkMte3VEvent);

        GlobalTensor<NegInfTypeT> valueOutGmTmp;
        valueOutGmTmp.SetGlobalBuffer((__gm__ NegInfTypeT *)valueOutGm.GetPhyAddr());
        GlobalTensor<NegInfTypeT> valueOut = valueOutGmTmp[paddingOffset];

        AscendC::InitGlobalMemory(valueOut, dealSize, (NegInfTypeT)(negInfBits));
        SetFlag<HardEvent::MTE3_V>(topkMte3VEvent);
        WaitFlag<HardEvent::MTE3_V>(topkMte3VEvent);
    }
}

} // namespace liVectorServiceCommon

#endif // LIGHTNING_INDEXER_V2_VECTOR_SERVICE_COMMON_H
