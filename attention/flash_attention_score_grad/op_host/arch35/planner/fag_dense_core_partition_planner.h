/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include "../model/fag_route_model.h"

namespace optiling {
namespace fag {

struct DenseCorePartitionInput {
    int64_t b = 0;
    int64_t n2 = 0;
    int64_t g = 0;
    int64_t s1Outer = 0;
    int64_t s2Outer = 0;
    uint32_t availableCoreCount = 0;
};

class DenseCorePartitionPlanner {
public:
    static CorePartitionOutcome TryBuild(VariantId variantId, const DenseCorePartitionInput &input)
    {
        if (input.availableCoreCount == 0 || input.b <= 0 || input.n2 <= 0 || input.g <= 0 || input.s1Outer <= 0 ||
            input.s2Outer <= 0) {
            return Reject(variantId, CandidateRejectReason::CORE_PARTITION_UNAVAILABLE, CandidateField::CORE_PARTITION,
                          0, 1);
        }

        switch (variantId) {
            case VariantId::BN2S2: {
                const int64_t groupedWork = input.b * input.n2 * input.g * input.s2Outer;
                return BuildGroupedPlan(variantId, CorePartitionMode::FUSED_BN2GS2, groupedWork, input.s1Outer,
                                        input.availableCoreCount);
            }
            case VariantId::BN2_MULTI_BLOCK: {
                const int64_t groupedWork = input.b * input.n2 * input.g;
                const int64_t workPerGroup = input.s1Outer * input.s2Outer;
                return BuildGroupedPlan(variantId, CorePartitionMode::FUSED_BN2G, groupedWork, workPerGroup,
                                        input.availableCoreCount);
            }
            case VariantId::BN2:
            case VariantId::BN2GS1S2: {
                const int64_t totalWork = input.b * input.n2 * input.g * input.s1Outer * input.s2Outer;
                return BuildGroupedPlan(variantId, CorePartitionMode::FUSED_BN2GS1S2, totalWork, 1,
                                        input.availableCoreCount);
            }
            default:
                return Reject(variantId, CandidateRejectReason::CORE_PARTITION_UNAVAILABLE,
                              CandidateField::CORE_PARTITION, 0, 1);
        }
    }

private:
    static CorePartitionOutcome BuildGroupedPlan(VariantId variantId, CorePartitionMode mode, int64_t groupedWork,
                                                 int64_t workPerGroup, uint32_t availableCoreCount)
    {
        if (groupedWork <= 0 || workPerGroup <= 0) {
            return Reject(variantId, CandidateRejectReason::CORE_PARTITION_UNAVAILABLE, CandidateField::CORE_PARTITION,
                          0, 1);
        }

        const int64_t coreCount = static_cast<int64_t>(availableCoreCount);
        const int64_t groupsPerCore = groupedWork / coreCount + static_cast<int64_t>(groupedWork % coreCount != 0);
        const int64_t activeCoreCount =
            groupedWork / groupsPerCore + static_cast<int64_t>(groupedWork % groupsPerCore != 0);
        if (activeCoreCount > static_cast<int64_t>(FAG_CORE_PARTITION_RANGE_CAPACITY)) {
            return Reject(variantId, CandidateRejectReason::CORE_RANGE_CAPACITY_EXCEEDED,
                          CandidateField::CORE_RANGE_CAPACITY, activeCoreCount,
                          static_cast<int64_t>(FAG_CORE_PARTITION_RANGE_CAPACITY));
        }

        CorePartitionOutcome outcome;
        outcome.accepted = true;
        CorePartitionPlan &plan = outcome.plan;
        plan.mode = mode;
        plan.activeCoreCount = static_cast<uint32_t>(activeCoreCount);
        plan.blockFactor = groupsPerCore * workPerGroup;
        plan.totalWork = groupedWork * workPerGroup;
        plan.maxCoreWork = static_cast<uint64_t>(plan.blockFactor);
        for (uint32_t coreIndex = 0; coreIndex < plan.activeCoreCount; ++coreIndex) {
            WorkRange &range = plan.ranges[coreIndex];
            range.begin = plan.blockFactor * coreIndex;
            const int64_t rangeEnd = plan.blockFactor * (coreIndex + 1);
            range.end = rangeEnd < plan.totalWork ? rangeEnd : plan.totalWork;
        }
        return outcome;
    }

    static CorePartitionOutcome Reject(VariantId variantId, CandidateRejectReason reason, CandidateField field,
                                       int64_t actual, int64_t expected)
    {
        CorePartitionOutcome outcome;
        outcome.rejected.variantId = variantId;
        outcome.rejected.stage = CandidateRejectStage::CORE_PARTITION;
        outcome.rejected.reason = reason;
        outcome.rejected.field = field;
        outcome.rejected.actual = actual;
        outcome.rejected.expected = expected;
        return outcome;
    }
};

} // namespace fag
} // namespace optiling
