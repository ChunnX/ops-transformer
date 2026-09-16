/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef LIGHTNING_INDEXER_V2_TND_PADDING_H
#define LIGHTNING_INDEXER_V2_TND_PADDING_H

#include "kernel_operator.h"

namespace LIV2Common {

template <typename ValueBits, typename ValueOutT, typename RunInfoT>
__aicore__ inline void DoTndPadding(const RunInfoT &runInfo, uint64_t topk, bool returnValue,
    ValueBits negativeInfinity, uint32_t kHeadNum, int32_t invalidIdx, uint32_t mte3VEvent,
    AscendC::GlobalTensor<int32_t> &indiceOutGm, AscendC::GlobalTensor<ValueOutT> &valueOutGm)
{
    uint32_t paddingLen = runInfo.curCuSeqlensQ - runInfo.curSequsedQ;
    uint64_t paddingOffset = runInfo.indiceOutOffset + runInfo.curSequsedQ * kHeadNum * topk;
    uint64_t dealSize = paddingLen * kHeadNum * topk;
    AscendC::GlobalTensor<int32_t> indiceOut = indiceOutGm[paddingOffset];
    AscendC::InitGlobalMemory(indiceOut, dealSize, invalidIdx);
    if (returnValue) {
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(mte3VEvent);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(mte3VEvent);
        AscendC::GlobalTensor<ValueBits> valueOutGmTmp;
        valueOutGmTmp.SetGlobalBuffer((__gm__ ValueBits *)valueOutGm.GetPhyAddr());
        AscendC::GlobalTensor<ValueBits> valueOut = valueOutGmTmp[paddingOffset];
        AscendC::InitGlobalMemory(valueOut, dealSize, negativeInfinity);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(mte3VEvent);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(mte3VEvent);
    }
}

} // namespace LIV2Common

#endif // LIGHTNING_INDEXER_V2_TND_PADDING_H
