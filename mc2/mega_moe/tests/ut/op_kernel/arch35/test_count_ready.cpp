/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <vector>
#include "../../../../op_kernel/arch35/common/mega_moe_count_ready.h"

namespace {
constexpr uint32_t CANARY = 0x5A123456U;
constexpr uint32_t MAX_COUNTS = 2048U;
constexpr uint32_t LANES = AscendC::Reg::LANES;
uint32_t Payload(uint32_t i)
{
    return i % 3 == 0 ? 0U : (i % 3 == 1 ? 0x00FFFFFFU : i);
}
void CheckSnapshot(uint32_t size, uint32_t epoch, int32_t badSlot)
{
    std::vector<uint32_t> counts(size + LANES, CANARY);
    std::array<uint32_t, LANES> result;
    result.fill(CANARY);
    for (uint32_t i = 0; i < size; ++i) {
        counts[i] = (epoch << 24U) | Payload(i);
    }
    if (badSlot >= 0)
        counts[badSlot] ^= 1U << 24U;
    MegaMoeImpl::CheckMteCountEpochsVF(counts.data(), result.data(), size, epoch);
    assert(result[0] == (badSlot >= 0 ? 1U : 0U));
    for (uint32_t i = 0; i < size; ++i)
        assert(counts[i] == Payload(i));
    for (uint32_t i = size; i < counts.size(); ++i)
        assert(counts[i] == CANARY);
    for (uint32_t i = 1; i < result.size(); ++i)
        assert(result[i] == CANARY);
}
void CheckPolling(uint32_t size, uint32_t epoch, uint32_t rounds)
{
    std::vector<int32_t> gm(size + LANES, static_cast<int32_t>(CANARY));
    std::vector<int32_t> ub(size + LANES, static_cast<int32_t>(CANARY));
    std::array<int32_t, LANES> result;
    result.fill(static_cast<int32_t>(CANARY));
    AscendC::pipeline = {};
    AscendC::pipeline.resultAddress = result.data();
    AscendC::pipeline.beforeCopy = [&](uint32_t round) {
        for (uint32_t i = 0; i < size; ++i) {
            auto tag = round == rounds || i < (round - 1) * size / rounds ? epoch : (epoch ^ 1U);
            gm[i] = static_cast<int32_t>((tag << 24U) | Payload(i));
        }
    };
    MegaMoeImpl::WaitForMteCountTable({gm.data()}, {ub.data()}, {result.data()}, size, epoch);
    auto &pipeline = AscendC::pipeline;
    assert(pipeline.copies == rounds && pipeline.scalarReads == rounds);
    assert(pipeline.copiedBytes == static_cast<uint64_t>(size) * sizeof(int32_t) * rounds);
    assert(!pipeline.mte2 && !pipeline.vector);
    assert(std::none_of(pipeline.eventSet.begin(), pipeline.eventSet.end(), [](bool set) { return set; }));
    if (rounds == 1)
        assert(pipeline.clockReads == 0);
    for (uint32_t i = 0; i < size; ++i) {
        assert(static_cast<uint32_t>(ub[i]) == Payload(i));
        assert(static_cast<uint32_t>(gm[i]) == ((epoch << 24U) | Payload(i)));
    }
    for (uint32_t i = size; i < ub.size(); ++i)
        assert(static_cast<uint32_t>(ub[i]) == CANARY);
    for (uint32_t i = 1; i < result.size(); ++i)
        assert(static_cast<uint32_t>(result[i]) == CANARY);
}
} // namespace

int main()
{
    if constexpr (g_coreType == AscendC::AIC) {
        MegaMoeImpl::WaitForMteCountTable({nullptr}, {nullptr}, {nullptr}, MAX_COUNTS, 0x80U);
        assert(AscendC::pipeline.copies == 0 && AscendC::pipeline.scalarReads == 0);
        std::cout << "count ready: AIC performs no accesses\n";
        return 0;
    }
    uint32_t snapshots = 0, polls = 0;
    // Every supported table length, including all partial-vector tails and the 8 KiB maximum.
    for (uint32_t size = 1; size <= MAX_COUNTS; ++size) {
        for (uint32_t epoch : {0x80U, 0xFFU}) {
            CheckSnapshot(size, epoch, -1);
            CheckSnapshot(size, epoch, static_cast<int32_t>(size - 1));
            snapshots += 2;
        }
    }
    // A single stale slot anywhere must survive accumulation across later vectors.
    for (uint32_t bad = 0; bad < MAX_COUNTS; ++bad) {
        CheckSnapshot(MAX_COUNTS, 0x81U, static_cast<int32_t>(bad));
        ++snapshots;
    }
    for (uint32_t epoch = 0x80U; epoch <= 0xFFU; ++epoch) {
        for (uint32_t size : {1U, 7U, 63U, 64U, 65U, 127U, 128U, 129U, 2047U, MAX_COUNTS}) {
            CheckPolling(size, epoch, 1);
            CheckPolling(size, epoch, 3);
            polls += 2;
        }
    }
    std::cout << "count ready: " << snapshots << " snapshots, " << polls << " polling cases passed\n";
}
