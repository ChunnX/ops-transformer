/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MEGA_MOE_SHARED_EXPERT_INPUT_H
#define MEGA_MOE_SHARED_EXPERT_INPUT_H

#include "kernel_operator.h"

namespace MegaMoeImpl {

// 共享输入保留独立的 data/scale 布局，直接消费量化 UB，不再从通信 GM 读回。
// 调用方保证 V_MTE3 已完成，并在全部目标搬运完成之前保持量化 UB 不变。
// 模板类型描述共享量化输出；当前与 MoE 相同，未来可接入独立量化结果而不改变共享布局。
template <typename ActivationType, typename QuantScaleType>
__aicore__ inline void StoreSharedExpertInputToken(GM_ADDR dataBase, GM_ADDR scaleBase, uint32_t tokenIndex,
                                                   uint32_t dataElementCount, uint32_t scaleElementCount,
                                                   uint32_t scaleOffsetInUbBytes,
                                                   const AscendC::LocalTensor<ActivationType> &quantOutput)
{
    if constexpr (g_coreType == AscendC::AIC) {
        return;
    }
    AscendC::GlobalTensor<ActivationType> dataGm;
    AscendC::GlobalTensor<QuantScaleType> scaleGm;
    dataGm.SetGlobalBuffer(reinterpret_cast<__gm__ ActivationType *>(dataBase));
    scaleGm.SetGlobalBuffer(reinterpret_cast<__gm__ QuantScaleType *>(scaleBase));
    auto scaleUb = quantOutput.template ReinterpretCast<uint8_t>()[scaleOffsetInUbBytes]
                       .template ReinterpretCast<QuantScaleType>();
    AscendC::DataCopyPad(dataGm[static_cast<uint64_t>(tokenIndex) * dataElementCount], quantOutput,
                         {1U, dataElementCount * static_cast<uint32_t>(sizeof(ActivationType)), 0U, 0U, 0U});
    AscendC::DataCopyPad(scaleGm[static_cast<uint64_t>(tokenIndex) * scaleElementCount], scaleUb,
                         {1U, scaleElementCount * static_cast<uint32_t>(sizeof(QuantScaleType)), 0U, 0U, 0U});
}

} // namespace MegaMoeImpl

#endif // MEGA_MOE_SHARED_EXPERT_INPUT_H
