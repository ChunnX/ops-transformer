/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MEGA_MOE_COUNT_READY_H
#define MEGA_MOE_COUNT_READY_H

#include "kernel_operator.h"
#include "mega_moe_constants.h"

namespace MegaMoeImpl {

// 仅用于 MTE 输入 count 整表重试，不改变其他 GM 状态轮询的退避间隔。
constexpr int64_t MTE_COUNT_POLL_BACKOFF_CYCLES = 500;

// 批量检查 epoch，并在同一份 UB 快照中掩出 count。只有 pending[0] == 0 时才可消费快照。
// pending 的各 lane 累积所有向量的失败结果，尾部无效 lane 不参与比较或写回。
__simd_vf__ inline void CheckMteCountEpochsVF(__ubuf__ uint32_t *counts, __ubuf__ uint32_t *pending,
                                              uint32_t elementCount, uint32_t expectEpoch)
{
    constexpr uint32_t ELEMENTS_PER_VECTOR = AscendC::GetVecLen() / sizeof(uint32_t);
    AscendC::Reg::RegTensor<uint32_t> countReg;
    AscendC::Reg::RegTensor<uint32_t> epochReg;
    AscendC::Reg::RegTensor<uint32_t> pendingReg;
    AscendC::Reg::RegTensor<uint32_t> oneReg;
    AscendC::Reg::RegTensor<uint32_t> countMaskReg;
    AscendC::Reg::RegTensor<uint32_t> resultReg;
    AscendC::Reg::MaskReg fullMask = AscendC::Reg::CreateMask<uint32_t, AscendC::Reg::MaskPattern::ALL>();
    AscendC::Reg::MaskReg firstLaneMask = AscendC::Reg::CreateMask<uint32_t, AscendC::Reg::MaskPattern::VL1>();
    AscendC::Reg::Duplicate(pendingReg, 0U, fullMask);
    AscendC::Reg::Duplicate(oneReg, 1U, fullMask);
    AscendC::Reg::Duplicate(countMaskReg, 0x00FFFFFFU, fullMask);
    for (uint32_t offset = 0U; offset < elementCount; offset += ELEMENTS_PER_VECTOR) {
        uint32_t validCount = elementCount - offset;
        AscendC::Reg::MaskReg activeMask = AscendC::Reg::UpdateMask<uint32_t>(validCount);
        AscendC::Reg::MaskReg mismatchMask;
        AscendC::Reg::LoadAlign(countReg, counts + offset);
        AscendC::Reg::ShiftRights(epochReg, countReg, static_cast<int16_t>(24), activeMask);
        AscendC::Reg::Compares<uint32_t, AscendC::CMPMODE::NE>(mismatchMask, epochReg, expectEpoch, activeMask);
        AscendC::Reg::Select(pendingReg, oneReg, pendingReg, mismatchMask);
        AscendC::Reg::And(countReg, countReg, countMaskReg, activeMask);
        AscendC::Reg::StoreAlign(counts + offset, countReg, activeMask);
    }
    AscendC::Reg::ReduceMax(resultReg, pendingReg, fullMask);
    AscendC::Reg::StoreAlign(pending, resultReg, firstLaneMask);
}

// counts 为连续 count 表，pending 为独立、32B 对齐的 UB 暂存区（至少一个 int32）。
// 首次立即读取；失败后等待 500 cycle，再重读整表。成功快照直接交给后续 VF cumsum。
__aicore__ inline void WaitForMteCountTable(const AscendC::GlobalTensor<int32_t> &countGm,
                                            const AscendC::LocalTensor<int32_t> &counts,
                                            const AscendC::LocalTensor<int32_t> &pending, uint32_t elementCount,
                                            uint32_t expectEpoch)
{
    if constexpr (g_coreType == AscendC::AIC) {
        return;
    }
    for (;;) {
        // 包括重试时的 count 去 tag 写回，必须完成后才能由 MTE2 覆盖同一 UB。
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_EVENT_ID2);
        AscendC::DataCopyPad(counts, countGm, {1U, elementCount * static_cast<uint32_t>(sizeof(int32_t)), 0U, 0U, 0U},
                             {true, 0U, 0U, 0U});
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(SYNC_EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(SYNC_EVENT_ID2);
        asc_vf_call<CheckMteCountEpochsVF>(reinterpret_cast<__ubuf__ uint32_t *>(counts.GetPhyAddr()),
                                           reinterpret_cast<__ubuf__ uint32_t *>(pending.GetPhyAddr()), elementCount,
                                           expectEpoch);
        AscendC::SetFlag<AscendC::HardEvent::V_S>(SYNC_EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(SYNC_EVENT_ID2);
        if (pending.GetValue(0U) == 0) {
            return;
        }
        int64_t startCycle = AscendC::GetSystemCycle();
        while (AscendC::GetSystemCycle() - startCycle < MTE_COUNT_POLL_BACKOFF_CYCLES) {
        }
    }
}

} // namespace MegaMoeImpl

#endif // MEGA_MOE_COUNT_READY_H
