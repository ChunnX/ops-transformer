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

struct PlannedCandidateVerificationInput {
    bool corePartitionAvailable = false;
    bool hasInvalidRow = false;
    bool hasInvalidColumn = false;
    uint32_t activeCoreCount = 0;
    uint32_t availableCoreCount = 0;
    int64_t blockFactor = 0;
    uint64_t maxCoreWork = 0;
};

class PlannedCandidateVerifier {
public:
    static CandidateOutcome Verify(const CandidatePlan &candidate, const PlannedCandidateVerificationInput &input)
    {
        switch (candidate.variantId) {
            case VariantId::BN2S2:
                return VerifyBn2S2(candidate, input);
            case VariantId::BN2_MULTI_BLOCK:
                return VerifyBn2MultiBlock(candidate, input);
            default:
                return Accept(candidate, input);
        }
    }

private:
    static CandidateOutcome VerifyBn2S2(const CandidatePlan &candidate, const PlannedCandidateVerificationInput &input)
    {
        if (!input.corePartitionAvailable) {
            return Reject(candidate.variantId, CandidateRejectStage::CORE_PARTITION,
                          CandidateRejectReason::CORE_PARTITION_UNAVAILABLE, CandidateField::CORE_PARTITION, 0, 1);
        }
        if (input.activeCoreCount < input.availableCoreCount) {
            return Reject(candidate.variantId, CandidateRejectStage::CORE_PARTITION,
                          CandidateRejectReason::INSUFFICIENT_CORE_UTILIZATION, CandidateField::ACTIVE_CORE_COUNT,
                          input.activeCoreCount, input.availableCoreCount);
        }
        return Accept(candidate, input);
    }

    static CandidateOutcome VerifyBn2MultiBlock(const CandidatePlan &candidate,
                                                const PlannedCandidateVerificationInput &input)
    {
        if (input.hasInvalidRow || input.hasInvalidColumn) {
            const int64_t invalidBlockMask =
                static_cast<int64_t>(input.hasInvalidRow) | (static_cast<int64_t>(input.hasInvalidColumn) << 1);
            return Reject(candidate.variantId, CandidateRejectStage::SPARSE_PLANNING,
                          CandidateRejectReason::INVALID_BASIC_BLOCK, CandidateField::INVALID_BASIC_BLOCK,
                          invalidBlockMask, 0);
        }
        return Accept(candidate, input);
    }

    static CandidateOutcome Accept(const CandidatePlan &candidate, const PlannedCandidateVerificationInput &input)
    {
        CandidateOutcome outcome;
        outcome.accepted = true;
        outcome.plan = candidate;
        outcome.plan.cores.activeCoreCount = input.activeCoreCount;
        outcome.plan.cores.blockFactor = input.blockFactor;
        outcome.plan.cores.maxCoreWork = input.maxCoreWork;
        outcome.plan.metrics.activeAic = input.activeCoreCount;
        outcome.plan.metrics.maxCoreWork = input.maxCoreWork;
        return outcome;
    }

    static CandidateOutcome Reject(VariantId variantId, CandidateRejectStage stage, CandidateRejectReason reason,
                                   CandidateField field, int64_t actual, int64_t expected)
    {
        CandidateOutcome outcome;
        outcome.rejected.variantId = variantId;
        outcome.rejected.stage = stage;
        outcome.rejected.reason = reason;
        outcome.rejected.field = field;
        outcome.rejected.actual = actual;
        outcome.rejected.expected = expected;
        return outcome;
    }
};

} // namespace fag
} // namespace optiling
