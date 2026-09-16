/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MEGA_MOE_EXPERT_COUNT_H
#define MEGA_MOE_EXPERT_COUNT_H

#include "kernel_operator.h"
#include "../common/mega_moe_constants.h"
#include "../common/mega_moe_utils.h"
// 实验性 SIMT 实现按需编译，默认 SIMD 路径不引入 SIMT API 依赖。
#if defined(MEGA_MOE_ENABLE_COUNT_SIMT)
#include "simt_api/asc_simt.h"
#endif

namespace MegaMoeImpl {

// count 整表检查失败后的重试间隔。
constexpr int64_t COUNT_TABLE_POLL_BACKOFF_CYCLES = 500;

// 批量检查高 8 位轮次标签，并在同一份 UB 快照中保留低 24 位 count。
// expectedSyncRoundTag 使用 GetSyncRoundTag 返回的高 8 位标签；pending[0] 为零时才可消费快照。
// pending 的各 lane 累积所有向量的失败结果，尾部无效 lane 不参与比较或写回。
__simd_vf__ inline void CheckCountRoundTagsVF(__ubuf__ uint32_t *counts, __ubuf__ uint32_t *pending,
                                              uint32_t elementCount, uint32_t expectedSyncRoundTag)
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
        AscendC::Reg::Compares<uint32_t, AscendC::CMPMODE::NE>(mismatchMask, epochReg, expectedSyncRoundTag >> 24U,
                                                               activeMask);
        AscendC::Reg::Select(pendingReg, oneReg, pendingReg, mismatchMask);
        AscendC::Reg::And(countReg, countReg, countMaskReg, activeMask);
        AscendC::Reg::StoreAlign(counts + offset, countReg, activeMask);
    }
    AscendC::Reg::Reduce<AscendC::Reg::ReduceType::MAX>(resultReg, pendingReg, fullMask);
    AscendC::Reg::StoreAlign(pending, resultReg, firstLaneMask);
}

// counts 为连续 count 表，pending 为独立、32B 对齐的 UB 暂存区（至少一个 int32）。
// 首次立即读取；失败后等待 500 cycle，再重读整表。成功快照直接交给后续 VF cumsum。
// 调用方保证在 AIV 上执行，且 elementCount > 0。
__aicore__ inline void WaitForCountTable(const AscendC::GlobalTensor<int32_t> &countGm,
                                         const AscendC::LocalTensor<int32_t> &counts,
                                         const AscendC::LocalTensor<int32_t> &pending, uint32_t elementCount,
                                         uint32_t expectedSyncRoundTag)
{
    for (;;) {
        // 包括重试时的 count 去 tag 写回，必须完成后才能由 MTE2 覆盖同一 UB。
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_EVENT_ID2);
        AscendC::DataCopyPad(counts, countGm, {1U, elementCount * static_cast<uint32_t>(sizeof(int32_t)), 0U, 0U, 0U},
                             {true, 0U, 0U, 0U});
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(SYNC_EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(SYNC_EVENT_ID2);
        asc_vf_call<CheckCountRoundTagsVF>(reinterpret_cast<__ubuf__ uint32_t *>(counts.GetPhyAddr()),
                                           reinterpret_cast<__ubuf__ uint32_t *>(pending.GetPhyAddr()), elementCount,
                                           expectedSyncRoundTag);
        AscendC::SetFlag<AscendC::HardEvent::V_S>(SYNC_EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(SYNC_EVENT_ID2);
        if (pending.GetValue(0U) == 0) {
            return;
        }
        int64_t startCycle = AscendC::GetSystemCycle();
        while (AscendC::GetSystemCycle() - startCycle < COUNT_TABLE_POLL_BACKOFF_CYCLES) {
        }
    }
}

#if defined(MEGA_MOE_ENABLE_COUNT_SIMT)
// 按线程编号步进处理 count；当前项就绪后写入 UB，再处理下一项。
template <uint32_t ThreadCount>
__simt_vf__ __launch_bounds__(ThreadCount) inline void WaitForCountTableSimtVF(__gm__ int32_t *countGm,
                                                                               __ubuf__ int32_t *counts,
                                                                               uint32_t elementCount,
                                                                               uint32_t expectedSyncRoundTag)
{
    for (uint32_t index = threadIdx.x; index < elementCount; index += blockDim.x) {
        uint32_t value;
        // ldcg 绕过 SIMT Data Cache；仅重读尚未匹配本轮标签的当前项。
        while (((value = static_cast<uint32_t>(asc_ldcg(countGm + index))) & 0xFF000000U) != expectedSyncRoundTag) {
            asc_nop(); // 短暂退避，降低 GM 轮询频率。
        }
        counts[index] = static_cast<int32_t>(value & 0x00FFFFFFU);
    }
}

// SIMT 实验入口，直接将就绪 count 写入 UB，无需整表检查结果暂存区。
// 启用前须为 SIMT Data Cache 预留至少 32 KiB，并核对整个 kernel 的 UB 布局及编译配置。
// 调用处替换为 WaitForCountTableSimt<>(...)；默认最大 1024 线程。
// 调用方保证在 AIV 上执行，且 elementCount > 0。
template <uint32_t ThreadCount = 1024U>
__aicore__ inline void WaitForCountTableSimt(__gm__ int32_t *countGm, __ubuf__ int32_t *counts, uint32_t elementCount,
                                             uint32_t expectedSyncRoundTag)
{
    // launch bounds 决定寄存器预算；实际线程数按表长向上对齐到一个 warp，最大为 ThreadCount。
    const uint32_t alignedThreadCount = (elementCount + 31U) / 32U * 32U;
    const uint32_t threadCount = alignedThreadCount < ThreadCount ? alignedThreadCount : ThreadCount;
    // 接上共享计算后的 MTE3_MTE2，确保旧 UB 数据搬出后 SIMT 才能覆盖；也等待先前的 MTE2 写入。
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(SYNC_EVENT_ID2);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(SYNC_EVENT_ID2);
    asc_vf_call<WaitForCountTableSimtVF<ThreadCount>>(dim3(threadCount), countGm, counts, elementCount,
                                                      expectedSyncRoundTag);
}
#endif

/*
 * MegaMoe count 表的融合版本：先复用 int32 prefix-scan，再在同一次 VF 中提取每个专家的累计尾值
 * 并作差。相比先调用通用前缀和、再启动第二个 VF，融合实现省去一次 asc_vf_call 和外部流水同步。
 */
__simd_vf__ inline void ComputeExpertCountTablesVF(__ubuf__ int32_t *count, __ubuf__ int32_t *expertCounts,
                                                   uint32_t elementCount, uint32_t expertCount, uint32_t worldSize,
                                                   uint32_t maxOutputSize)
{
    InclusivePrefixSumInt32(count, elementCount);

    // prefix 已写回 UB；后续从同一区域 Gather 前建立 VEC_STORE -> VEC_LOAD 可见性。
    AscendC::Reg::LocalMemBar<AscendC::Reg::MemType::VEC_STORE, AscendC::Reg::MemType::VEC_LOAD>();

    constexpr uint32_t ELEMENTS_PER_VECTOR = GetVecLen() / sizeof(int32_t);
    AscendC::Reg::RegTensor<int32_t> expertEndPrefixReg;
    AscendC::Reg::RegTensor<int32_t> previousEndPrefixReg;
    AscendC::Reg::RegTensor<int32_t> expertCountReg;
    AscendC::Reg::RegTensor<int32_t> previousVectorEndReg;
    AscendC::Reg::RegTensor<int32_t> maxOutputSizeReg;
    AscendC::Reg::RegTensor<int32_t> laneIndexReg;
    AscendC::Reg::RegTensor<int32_t> gatherIndexReg;
    AscendC::Reg::RegTensor<int32_t> previousLaneIndexReg;
    AscendC::Reg::MaskReg fullMask = AscendC::Reg::CreateMask<int32_t, AscendC::Reg::MaskPattern::ALL>();
    AscendC::Reg::MaskReg firstLaneMask = AscendC::Reg::CreateMask<int32_t, AscendC::Reg::MaskPattern::VL1>();

    int32_t maxOutputSizeInt32 = static_cast<int32_t>(maxOutputSize);
    AscendC::Reg::Arange(laneIndexReg, 0);
    AscendC::Reg::Duplicate(previousVectorEndReg, 0, fullMask);
    AscendC::Reg::Duplicate(maxOutputSizeReg, maxOutputSizeInt32, fullMask);
    for (uint32_t expertOffset = 0U; expertOffset < expertCount; expertOffset += ELEMENTS_PER_VECTOR) {
        uint32_t validCount = expertCount - expertOffset;
        validCount = validCount < ELEMENTS_PER_VECTOR ? validCount : ELEMENTS_PER_VECTOR;
        uint32_t maskCount = validCount;
        AscendC::Reg::MaskReg activeMask = AscendC::Reg::UpdateMask<int32_t>(maskCount);

        AscendC::Reg::Adds(gatherIndexReg, laneIndexReg, static_cast<int32_t>(expertOffset), fullMask);
        AscendC::Reg::Muls(gatherIndexReg, gatherIndexReg, static_cast<int32_t>(worldSize), fullMask);
        AscendC::Reg::Adds(gatherIndexReg, gatherIndexReg, static_cast<int32_t>(worldSize - 1U), fullMask);
        AscendC::Reg::Gather(expertEndPrefixReg, count,
                             reinterpret_cast<AscendC::Reg::RegTensor<uint32_t> &>(gatherIndexReg), activeMask);

        AscendC::Reg::Adds(previousLaneIndexReg, laneIndexReg, -1, fullMask);
        AscendC::Reg::Maxs(previousLaneIndexReg, previousLaneIndexReg, 0, fullMask);
        AscendC::Reg::Gather(previousEndPrefixReg, expertEndPrefixReg,
                             reinterpret_cast<AscendC::Reg::RegTensor<uint32_t> &>(previousLaneIndexReg));
        AscendC::Reg::Select(previousEndPrefixReg, previousVectorEndReg, previousEndPrefixReg, firstLaneMask);
        AscendC::Reg::MaskReg overflowMask;
        AscendC::Reg::Compares<int32_t, AscendC::CMPMODE::GE>(overflowMask, expertEndPrefixReg, maxOutputSizeInt32,
                                                              activeMask);
        AscendC::Reg::Select(expertCountReg, maxOutputSizeReg, expertEndPrefixReg, overflowMask);
        AscendC::Reg::Compares<int32_t, AscendC::CMPMODE::GE>(overflowMask, previousEndPrefixReg, maxOutputSizeInt32,
                                                              activeMask);
        AscendC::Reg::Select(previousEndPrefixReg, maxOutputSizeReg, previousEndPrefixReg, overflowMask);
        AscendC::Reg::Sub(expertCountReg, expertCountReg, previousEndPrefixReg, activeMask);
        AscendC::Reg::StoreAlign(expertCounts + expertOffset, expertCountReg, activeMask);

        AscendC::Reg::Duplicate(previousLaneIndexReg, static_cast<int32_t>(validCount - 1U), fullMask);
        AscendC::Reg::Gather(previousVectorEndReg, expertEndPrefixReg,
                             reinterpret_cast<AscendC::Reg::RegTensor<uint32_t> &>(previousLaneIndexReg));
    }
}

__aicore__ inline void ComputeExpertCountTables(LocalTensor<int32_t> countTensor,
                                                LocalTensor<int32_t> expertCountTensor, uint32_t expertCount,
                                                uint32_t worldSize, uint32_t maxOutputSize)
{
    __ubuf__ int32_t *count = reinterpret_cast<__ubuf__ int32_t *>(countTensor.GetPhyAddr());
    __ubuf__ int32_t *expertCounts = reinterpret_cast<__ubuf__ int32_t *>(expertCountTensor.GetPhyAddr());
    asc_vf_call<ComputeExpertCountTablesVF>(count, expertCounts, expertCount * worldSize, expertCount, worldSize,
                                            maxOutputSize);
}

} // namespace MegaMoeImpl

#endif // MEGA_MOE_EXPERT_COUNT_H
