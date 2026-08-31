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

#include "../candidate/fag_initial_variant_candidates.h"

namespace optiling {
namespace fag {

struct LegacyOutputFormatPolicyInput {
    bool nzKernelSupported = false;
    bool enableSwizzle = false;
    int64_t s1 = 0;
    int64_t s2 = 0;
    int64_t minimumSequenceLength = 0;
};

struct OutputFormatDecision {
    OutputFormatPlan plan;
    bool capabilityAccepted = false;
    bool performanceAccepted = false;
    RuleId winningRule = FAG_RULE_LEGACY_DEFAULT;
};

class LegacyPolicyV1 {
public:
    static RuleId GetVariantRule(VariantId variantId)
    {
        switch (variantId) {
            case VariantId::BN2_MULTI_BLOCK:
                return FAG_RULE_LEGACY_INITIAL_BN2_MULTI_BLOCK;
            case VariantId::BN2:
                return FAG_RULE_LEGACY_INITIAL_BN2;
            case VariantId::BN2S2:
                return FAG_RULE_LEGACY_INITIAL_BN2S2;
            case VariantId::BN2GS1S2:
                return FAG_RULE_LEGACY_INITIAL_BN2GS1S2;
            default:
                return FAG_RULE_LEGACY_DEFAULT;
        }
    }

    static SelectionResult SelectInitialVariant(const InitialVariantCandidateSet &candidates)
    {
        const VariantId legacyPriority[FAG_INITIAL_VARIANT_CANDIDATE_CAPACITY] = {
            VariantId::BN2_MULTI_BLOCK, VariantId::BN2, VariantId::BN2S2, VariantId::BN2GS1S2};
        SelectionResult result;
        for (const VariantId variantId : legacyPriority) {
            const CandidateOutcome *outcome = candidates.Find(variantId);
            if (outcome == nullptr) {
                continue;
            }
            if (!outcome->accepted) {
                result.RecordRejection(outcome->rejected);
            } else if (result.selected.variantId == VariantId::UNKNOWN) {
                result.selected = outcome->plan;
                result.winningRule = GetVariantRule(variantId);
            }
        }
        return result;
    }

    static SelectionResult SelectFallbackAfterRejection(const SelectionResult &current,
                                                        const CandidateRejected &rejected,
                                                        const CandidatePlan &fallback)
    {
        SelectionResult result = current;
        result.RecordRejection(rejected);
        result.selected = fallback;
        result.winningRule = GetVariantRule(fallback.variantId);
        return result;
    }

    static VariantId ResolveVariant(bool isBn2, bool isBn2MultiBlock, bool isBn2S2)
    {
        if (isBn2 && isBn2MultiBlock) {
            return VariantId::BN2_MULTI_BLOCK;
        }
        if (isBn2) {
            return VariantId::BN2;
        }
        if (isBn2S2) {
            return VariantId::BN2S2;
        }
        return VariantId::BN2GS1S2;
    }

    static OutputFormatDecision SelectOutputFormat(const LegacyOutputFormatPolicyInput &input)
    {
        OutputFormatDecision decision;
        decision.capabilityAccepted = input.nzKernelSupported;
        decision.performanceAccepted =
            input.enableSwizzle && input.s1 >= input.minimumSequenceLength && input.s2 >= input.minimumSequenceLength;
        if (decision.capabilityAccepted && decision.performanceAccepted) {
            decision.plan.format = OutputFormat::NZ;
            decision.winningRule = FAG_RULE_LEGACY_NZ_LARGE_SEQUENCE;
        }
        return decision;
    }
};

} // namespace fag
} // namespace optiling
