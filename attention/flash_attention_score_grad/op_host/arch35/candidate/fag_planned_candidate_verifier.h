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

struct PlannedCandidateObservation {
    CorePartitionOutcome corePartition;
    bool hasInvalidRow = false;
    bool hasInvalidColumn = false;
    uint32_t availableCoreCount = 0;
};

struct PlannedCandidateExecutionResult {
    bool succeeded = false;
    PlannedCandidateObservation observation;
};

enum class PlannedCandidateAction : uint8_t {
    ACCEPT = 0,
    FALLBACK = 1,
    FAIL = 2,
};

struct PlannedCandidateDecision {
    PlannedCandidateAction action = PlannedCandidateAction::FAIL;
    CandidateOutcome candidate;
};

class PlannedCandidateVerifier {
public:
    static CandidateOutcome Verify(const CandidatePlan &candidate, const PlannedCandidateObservation &observation)
    {
        switch (candidate.variantId) {
            case VariantId::BN2S2:
                return VerifyBn2S2(candidate, observation);
            case VariantId::BN2_MULTI_BLOCK:
                return VerifyBn2MultiBlock(candidate, observation);
            case VariantId::BN2:
            case VariantId::BN2GS1S2:
                return Accept(candidate, observation);
            default:
                return Reject(candidate.variantId, CandidateRejectStage::CORE_PARTITION,
                              CandidateRejectReason::CORE_PARTITION_UNAVAILABLE, CandidateField::CORE_PARTITION, 0, 1);
        }
    }

private:
    static CandidateOutcome VerifyBn2S2(const CandidatePlan &candidate, const PlannedCandidateObservation &observation)
    {
        if (!observation.corePartition.accepted) {
            if (observation.corePartition.rejected.reason != CandidateRejectReason::NONE &&
                observation.corePartition.rejected.variantId == candidate.variantId) {
                CandidateOutcome outcome;
                outcome.rejected = observation.corePartition.rejected;
                return outcome;
            }
            return Reject(candidate.variantId, CandidateRejectStage::CORE_PARTITION,
                          CandidateRejectReason::CORE_PARTITION_UNAVAILABLE, CandidateField::CORE_PARTITION, 0, 1);
        }
        if (observation.corePartition.plan.activeCoreCount < observation.availableCoreCount) {
            return Reject(candidate.variantId, CandidateRejectStage::CORE_PARTITION,
                          CandidateRejectReason::INSUFFICIENT_CORE_UTILIZATION, CandidateField::ACTIVE_CORE_COUNT,
                          observation.corePartition.plan.activeCoreCount, observation.availableCoreCount);
        }
        return Accept(candidate, observation);
    }

    static CandidateOutcome VerifyBn2MultiBlock(const CandidatePlan &candidate,
                                                const PlannedCandidateObservation &observation)
    {
        if (observation.hasInvalidRow || observation.hasInvalidColumn) {
            const int64_t invalidBlockMask = static_cast<int64_t>(observation.hasInvalidRow) |
                                             (static_cast<int64_t>(observation.hasInvalidColumn) << 1);
            return Reject(candidate.variantId, CandidateRejectStage::SPARSE_PLANNING,
                          CandidateRejectReason::INVALID_BASIC_BLOCK, CandidateField::INVALID_BASIC_BLOCK,
                          invalidBlockMask, 0);
        }
        return Accept(candidate, observation);
    }

    static CandidateOutcome Accept(const CandidatePlan &candidate, const PlannedCandidateObservation &observation)
    {
        CandidateOutcome outcome;
        outcome.accepted = true;
        outcome.plan = candidate;
        if (observation.corePartition.accepted) {
            outcome.plan.cores = observation.corePartition.plan;
            outcome.plan.metrics.activeAic = observation.corePartition.plan.activeCoreCount;
            outcome.plan.metrics.maxCoreWork = observation.corePartition.plan.maxCoreWork;
        }
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

class PlannedCandidateResolver {
public:
    static PlannedCandidateDecision Resolve(const CandidatePlan &candidate,
                                            const PlannedCandidateExecutionResult &execution)
    {
        PlannedCandidateDecision decision;
        decision.candidate = PlannedCandidateVerifier::Verify(candidate, execution.observation);
        if (!decision.candidate.accepted) {
            decision.action = PlannedCandidateAction::FALLBACK;
        } else if (execution.succeeded) {
            decision.action = PlannedCandidateAction::ACCEPT;
        }
        return decision;
    }
};

} // namespace fag
} // namespace optiling
