/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>

#include "../../../../op_host/arch35/candidate/fag_kernel_capability.h"
#include "../../../../op_host/arch35/candidate/fag_planned_candidate_verifier.h"
#include "../../../../op_host/arch35/model/fag_route_model.h"
#include "../../../../op_host/arch35/policy/fag_legacy_policy_v1.h"

namespace optiling {
namespace fag {
namespace {

InitialVariantLimits MakeInitialVariantLimits()
{
    InitialVariantLimits limits;
    limits.bn2MaxSequence = 128;
    limits.bn2MaxHeadDim = 512;
    limits.bn2MultiBlockMaxSequence = 640;
    limits.bn2MultiBlockBatchHeadLow = 128;
    limits.bn2MultiBlockBatchHeadHigh = 256;
    limits.bn2s2MaxSequence = 1024;
    limits.bn2s2WriteUbMaxHeadDim = 128;
    limits.bn2TndMaxHeadDim = 128;
    limits.bn2MultiBlockSequenceAlignment = 128;
    limits.bn2s2MinimumSequenceGap = 128;
    limits.bn2s2DropoutMaxHeadDim = 128;
    limits.tndExecutionLayout = 4;
    limits.prefixSparseMode = 5;
    limits.prefixCompressSparseMode = 6;
    limits.bn2SplitAxis = 1;
    limits.bn2s2SplitAxis = 5;
    limits.bn2gs1s2SplitAxis = 0;
    return limits;
}

InitialVariantCandidateInput MakeInitialVariantInput()
{
    InitialVariantCandidateInput input;
    input.b = 1;
    input.s1 = 64;
    input.s2 = 64;
    input.n1 = 64;
    input.n2 = 64;
    input.d = 64;
    input.d1 = 64;
    input.executionLayout = 1;
    input.queryTypeSupported = true;
    input.dropoutDisabled = true;
    return input;
}

struct LegacyInitialVariantDecision {
    VariantId variantId = VariantId::UNKNOWN;
    uint32_t executionLayout = 0;
    bool effectiveDeterministic = false;
};

struct LegacyPostPlanningDecision {
    VariantId variantId = VariantId::UNKNOWN;
    bool effectiveDeterministic = false;
};

LegacyPostPlanningDecision SelectWithFrozenLegacyPostPlanning(VariantId initialVariant, bool corePartitionAvailable,
                                                              uint32_t activeCoreCount, uint32_t availableCoreCount,
                                                              bool hasInvalidRow, bool hasInvalidColumn,
                                                              bool deterministicRequested)
{
    LegacyPostPlanningDecision decision;
    decision.variantId = initialVariant;
    decision.effectiveDeterministic = initialVariant == VariantId::BN2 || initialVariant == VariantId::BN2_MULTI_BLOCK ?
                                          false :
                                          deterministicRequested;
    const bool rejectBn2S2 =
        initialVariant == VariantId::BN2S2 && (!corePartitionAvailable || activeCoreCount < availableCoreCount);
    const bool rejectBn2MultiBlock =
        initialVariant == VariantId::BN2_MULTI_BLOCK && (hasInvalidRow || hasInvalidColumn);
    if (rejectBn2S2 || rejectBn2MultiBlock) {
        decision.variantId = VariantId::BN2GS1S2;
        decision.effectiveDeterministic = deterministicRequested;
    }
    return decision;
}

LegacyInitialVariantDecision SelectWithFrozenLegacyExpression(const InitialVariantCandidateInput &input,
                                                              const InitialVariantLimits &limits)
{
    bool effectiveDeterministic = input.deterministicRequested;
    bool isBn2 = input.s1 <= limits.bn2MaxSequence && input.s2 <= limits.bn2MaxSequence && input.n1 == input.n2 &&
                 input.d <= limits.bn2MaxHeadDim && input.queryTypeSupported && input.tailZeroCount == 0 &&
                 !input.sequenceContainsZero;
    const int64_t batchHeadProduct = input.b * input.n1;
    const bool batchHeadLimit =
        batchHeadProduct >= limits.bn2MultiBlockBatchHeadHigh ||
        (batchHeadProduct >= limits.bn2MultiBlockBatchHeadLow && limits.bn2MultiBlockSequenceAlignment > 0 &&
         input.s1 % limits.bn2MultiBlockSequenceAlignment == 0 &&
         input.s2 % limits.bn2MultiBlockSequenceAlignment == 0);
    const bool sparseLimit = batchHeadLimit && input.executionLayout != limits.tndExecutionLayout &&
                             input.sparseMode != limits.prefixSparseMode &&
                             input.sparseMode != limits.prefixCompressSparseMode;
    bool isBn2MultiBlock = sparseLimit && (input.s1 > limits.bn2MaxSequence || input.s2 > limits.bn2MaxSequence) &&
                           input.s1 <= limits.bn2MultiBlockMaxSequence && input.s2 <= limits.bn2MultiBlockMaxSequence &&
                           input.n1 == input.n2 && input.d <= limits.bn2MaxHeadDim && input.queryTypeSupported &&
                           input.d == input.d1 && !input.hasRope;
    isBn2 = isBn2MultiBlock || isBn2;
    if (isBn2 && !isBn2MultiBlock) {
        effectiveDeterministic = false;
        if ((input.executionLayout == limits.tndExecutionLayout && input.d > limits.bn2TndMaxHeadDim) ||
            input.dropMaskOuter) {
            isBn2 = false;
            effectiveDeterministic = input.deterministicRequested;
        }
    }
    if (isBn2MultiBlock) {
        effectiveDeterministic = false;
        if (input.dropMaskOuter) {
            isBn2 = false;
            isBn2MultiBlock = false;
            effectiveDeterministic = input.deterministicRequested;
        }
    }

    const bool nonTndRoute = input.s1 < input.s2 && input.s2 <= limits.bn2s2MaxSequence &&
                             input.s2 - input.s1 >= limits.bn2s2MinimumSequenceGap &&
                             input.d <= limits.bn2s2WriteUbMaxHeadDim && !input.isSparse && !effectiveDeterministic;
    const bool bn2s2Route =
        !input.hasRope && input.d <= limits.bn2MaxHeadDim &&
        (input.executionLayout == limits.tndExecutionLayout || (input.allSequencesSame && !effectiveDeterministic) ||
         nonTndRoute) &&
        (input.dropoutDisabled || (input.d <= limits.bn2s2DropoutMaxHeadDim && input.dropoutEnabled)) &&
        input.n1 == input.n2 && input.queryTypeSupported;

    LegacyInitialVariantDecision decision;
    decision.executionLayout = input.executionLayout;
    decision.effectiveDeterministic = effectiveDeterministic;
    if (isBn2MultiBlock) {
        decision.variantId = VariantId::BN2_MULTI_BLOCK;
    } else if (isBn2) {
        decision.variantId = VariantId::BN2;
    } else if (bn2s2Route) {
        decision.variantId = VariantId::BN2S2;
        decision.executionLayout = input.allSequencesSame ? limits.tndExecutionLayout : input.executionLayout;
    } else {
        decision.variantId = VariantId::BN2GS1S2;
    }
    return decision;
}

TEST(FagRouteModel, ProblemFingerprintIsStableAndSensitiveToSourceFacts)
{
    ProblemFacts facts;
    facts.sourceLayout = SourceLayoutId::TND;
    facts.queryType = 1;
    facts.sparseMode = 3;
    facts.deterministicRequested = true;
    facts.hasRope = false;
    facts.b = 2;
    facts.s1 = 2048;
    facts.s2 = 4096;
    facts.n1 = 32;
    facts.n2 = 8;
    facts.g = 4;
    facts.d = 96;
    facts.d1 = 128;

    const uint64_t fingerprint = facts.Fingerprint();
    ProblemFacts sameFacts = facts;
    EXPECT_EQ(fingerprint, sameFacts.Fingerprint());

    sameFacts.sourceLayout = SourceLayoutId::SBH;
    EXPECT_NE(fingerprint, sameFacts.Fingerprint());
    sameFacts = facts;
    sameFacts.deterministicRequested = false;
    EXPECT_NE(fingerprint, sameFacts.Fingerprint());
}

TEST(FagRouteModel, TraceProtocolIdsAreExplicitAndStable)
{
    EXPECT_EQ(1U, FAG_ARCH35_LEGACY_POLICY_VERSION);
    EXPECT_EQ(4U, static_cast<uint32_t>(SourceLayoutId::TND));
    EXPECT_EQ(5U, static_cast<uint32_t>(SourceLayoutId::BSND));
    EXPECT_EQ(2U, static_cast<uint32_t>(VariantId::BN2_MULTI_BLOCK));
    EXPECT_EQ(500U, static_cast<uint32_t>(RouteReason::LEGACY_NZ_CAPABILITY_REJECTED));
    EXPECT_EQ(801U, static_cast<uint32_t>(RouteReason::FINAL_KEY_ENCODED));
    EXPECT_EQ(1001U, FAG_RULE_LEGACY_NZ_LARGE_SEQUENCE);
    EXPECT_EQ(1004U, static_cast<uint32_t>(CandidateRejectReason::SEQUENCE_CONTAINS_ZERO));
    EXPECT_EQ(1013U, static_cast<uint32_t>(CandidateRejectReason::CORE_PARTITION_UNAVAILABLE));
    EXPECT_EQ(1015U, static_cast<uint32_t>(CandidateRejectReason::INVALID_BASIC_BLOCK));
    EXPECT_EQ(1100U, FAG_RULE_LEGACY_INITIAL_BN2_MULTI_BLOCK);
    EXPECT_EQ(1103U, FAG_RULE_LEGACY_INITIAL_BN2GS1S2);
}

TEST(FagInitialVariantCandidates, BuildsEachVariantInIndependentScratchPlan)
{
    const InitialVariantLimits limits = MakeInitialVariantLimits();
    InitialVariantCandidateInput input = MakeInitialVariantInput();

    CandidateOutcome outcome = Bn2Strategy::TryBuild(input, limits);
    ASSERT_TRUE(outcome.accepted);
    EXPECT_EQ(static_cast<uint32_t>(VariantId::BN2), static_cast<uint32_t>(outcome.plan.variantId));
    EXPECT_FALSE(outcome.plan.effectiveDeterministic);

    input.b = 4;
    input.s1 = 256;
    input.s2 = 256;
    outcome = Bn2MultiBlockStrategy::TryBuild(input, limits);
    ASSERT_TRUE(outcome.accepted);
    EXPECT_EQ(static_cast<uint32_t>(VariantId::BN2_MULTI_BLOCK), static_cast<uint32_t>(outcome.plan.variantId));
    EXPECT_TRUE(outcome.plan.split.isMultiBlock);

    input = MakeInitialVariantInput();
    input.s2 = 192;
    outcome = Bn2S2Strategy::TryBuild(input, limits);
    ASSERT_TRUE(outcome.accepted);
    EXPECT_EQ(static_cast<uint32_t>(VariantId::BN2S2), static_cast<uint32_t>(outcome.plan.variantId));

    input = MakeInitialVariantInput();
    input.queryTypeSupported = false;
    outcome = Bn2Gs1S2Strategy::TryBuild(input, limits);
    ASSERT_TRUE(outcome.accepted);
    EXPECT_EQ(static_cast<uint32_t>(VariantId::BN2GS1S2), static_cast<uint32_t>(outcome.plan.variantId));
}

TEST(FagInitialVariantCandidates, ReturnsStableRejectReasonWithoutFallbackMutation)
{
    const InitialVariantLimits limits = MakeInitialVariantLimits();
    InitialVariantCandidateInput input = MakeInitialVariantInput();
    input.sequenceContainsZero = true;
    const CandidateOutcome bn2 = Bn2Strategy::TryBuild(input, limits);
    EXPECT_FALSE(bn2.accepted);
    EXPECT_EQ(static_cast<uint32_t>(VariantId::BN2), static_cast<uint32_t>(bn2.rejected.variantId));
    EXPECT_EQ(static_cast<uint32_t>(CandidateRejectReason::SEQUENCE_CONTAINS_ZERO),
              static_cast<uint32_t>(bn2.rejected.reason));
    EXPECT_TRUE(input.sequenceContainsZero);

    input = MakeInitialVariantInput();
    input.b = 4;
    input.s1 = 256;
    input.s2 = 256;
    input.dropMaskOuter = true;
    const CandidateOutcome multiBlock = Bn2MultiBlockStrategy::TryBuild(input, limits);
    EXPECT_FALSE(multiBlock.accepted);
    EXPECT_EQ(static_cast<uint32_t>(CandidateRejectReason::DROP_MASK_UNSUPPORTED),
              static_cast<uint32_t>(multiBlock.rejected.reason));
    EXPECT_TRUE(input.dropMaskOuter);
}

TEST(FagPlannedCandidateVerifier, RejectsBn2S2WhenCorePartitionIsUnavailable)
{
    CandidatePlan candidate;
    candidate.variantId = VariantId::BN2S2;
    PlannedCandidateVerificationInput input;
    const CandidateOutcome outcome = PlannedCandidateVerifier::Verify(candidate, input);

    EXPECT_FALSE(outcome.accepted);
    EXPECT_EQ(static_cast<uint32_t>(CandidateRejectStage::CORE_PARTITION),
              static_cast<uint32_t>(outcome.rejected.stage));
    EXPECT_EQ(static_cast<uint32_t>(CandidateRejectReason::CORE_PARTITION_UNAVAILABLE),
              static_cast<uint32_t>(outcome.rejected.reason));
    EXPECT_EQ(static_cast<uint32_t>(CandidateField::CORE_PARTITION), static_cast<uint32_t>(outcome.rejected.field));
}

TEST(FagPlannedCandidateVerifier, VerifiesBn2S2CoreUtilizationWithoutMutatingInputPlan)
{
    CandidatePlan candidate;
    candidate.variantId = VariantId::BN2S2;
    candidate.cores.activeCoreCount = 7;
    PlannedCandidateVerificationInput input;
    input.corePartitionAvailable = true;
    input.activeCoreCount = 31;
    input.availableCoreCount = 32;
    input.blockFactor = 64;
    input.maxCoreWork = 63;

    CandidateOutcome outcome = PlannedCandidateVerifier::Verify(candidate, input);
    EXPECT_FALSE(outcome.accepted);
    EXPECT_EQ(static_cast<uint32_t>(CandidateRejectReason::INSUFFICIENT_CORE_UTILIZATION),
              static_cast<uint32_t>(outcome.rejected.reason));
    EXPECT_EQ(31, outcome.rejected.actual);
    EXPECT_EQ(32, outcome.rejected.expected);
    EXPECT_EQ(7U, candidate.cores.activeCoreCount);

    input.activeCoreCount = input.availableCoreCount;
    outcome = PlannedCandidateVerifier::Verify(candidate, input);
    ASSERT_TRUE(outcome.accepted);
    EXPECT_EQ(32U, outcome.plan.cores.activeCoreCount);
    EXPECT_EQ(64, outcome.plan.cores.blockFactor);
    EXPECT_EQ(63U, outcome.plan.cores.maxCoreWork);
    EXPECT_EQ(7U, candidate.cores.activeCoreCount);
}

TEST(FagPlannedCandidateVerifier, RejectsBn2MultiBlockWithStableInvalidBlockMask)
{
    CandidatePlan candidate;
    candidate.variantId = VariantId::BN2_MULTI_BLOCK;
    PlannedCandidateVerificationInput input;
    input.corePartitionAvailable = true;
    input.hasInvalidRow = true;
    input.hasInvalidColumn = true;

    const CandidateOutcome outcome = PlannedCandidateVerifier::Verify(candidate, input);
    EXPECT_FALSE(outcome.accepted);
    EXPECT_EQ(static_cast<uint32_t>(CandidateRejectStage::SPARSE_PLANNING),
              static_cast<uint32_t>(outcome.rejected.stage));
    EXPECT_EQ(static_cast<uint32_t>(CandidateRejectReason::INVALID_BASIC_BLOCK),
              static_cast<uint32_t>(outcome.rejected.reason));
    EXPECT_EQ(3, outcome.rejected.actual);
    EXPECT_EQ(0, outcome.rejected.expected);
}

TEST(FagLegacyPolicyV1, FallbackSelectionPreservesCandidateRejectionHistory)
{
    SelectionResult current;
    current.selected.variantId = VariantId::BN2S2;
    CandidateRejected initialRejected;
    initialRejected.variantId = VariantId::BN2;
    initialRejected.reason = CandidateRejectReason::HEAD_DIM_UNSUPPORTED;
    current.RecordRejection(initialRejected);

    CandidateRejected plannedRejected;
    plannedRejected.variantId = VariantId::BN2S2;
    plannedRejected.stage = CandidateRejectStage::CORE_PARTITION;
    plannedRejected.reason = CandidateRejectReason::INSUFFICIENT_CORE_UTILIZATION;
    CandidatePlan fallback;
    fallback.variantId = VariantId::BN2GS1S2;

    const SelectionResult selected = LegacyPolicyV1::SelectFallbackAfterRejection(current, plannedRejected, fallback);
    EXPECT_EQ(static_cast<uint32_t>(VariantId::BN2GS1S2), static_cast<uint32_t>(selected.selected.variantId));
    EXPECT_EQ(FAG_RULE_LEGACY_INITIAL_BN2GS1S2, selected.winningRule);
    ASSERT_EQ(2U, selected.GetRejectionCount());
    EXPECT_EQ(static_cast<uint32_t>(VariantId::BN2), static_cast<uint32_t>(selected.GetRejection(0).variantId));
    EXPECT_EQ(static_cast<uint32_t>(VariantId::BN2S2), static_cast<uint32_t>(selected.GetRejection(1).variantId));
}

TEST(FagLegacyPolicyV1, PlannedCandidateVerificationMatchesFrozenLegacyFallback)
{
    const VariantId variants[] = {VariantId::BN2_MULTI_BLOCK, VariantId::BN2, VariantId::BN2S2, VariantId::BN2GS1S2};
    size_t variantMismatchCount = 0;
    size_t deterministicMismatchCount = 0;
    for (const VariantId variantId : variants) {
        for (uint32_t mask = 0; mask < 32; ++mask) {
            const bool corePartitionAvailable = (mask & 1U) != 0;
            const uint32_t activeCoreCount = (mask & 2U) == 0 ? 31U : 32U;
            const bool hasInvalidRow = (mask & 4U) != 0;
            const bool hasInvalidColumn = (mask & 8U) != 0;
            const bool deterministicRequested = (mask & 16U) != 0;
            const LegacyPostPlanningDecision expected =
                SelectWithFrozenLegacyPostPlanning(variantId, corePartitionAvailable, activeCoreCount, 32U,
                                                   hasInvalidRow, hasInvalidColumn, deterministicRequested);

            CandidatePlan initial;
            initial.variantId = variantId;
            initial.effectiveDeterministic =
                variantId == VariantId::BN2 || variantId == VariantId::BN2_MULTI_BLOCK ? false : deterministicRequested;
            PlannedCandidateVerificationInput input;
            input.corePartitionAvailable = corePartitionAvailable;
            input.activeCoreCount = activeCoreCount;
            input.availableCoreCount = 32U;
            input.hasInvalidRow = hasInvalidRow;
            input.hasInvalidColumn = hasInvalidColumn;
            const CandidateOutcome verified = PlannedCandidateVerifier::Verify(initial, input);

            SelectionResult actual;
            actual.selected = initial;
            if (!verified.accepted) {
                CandidatePlan fallback;
                fallback.variantId = VariantId::BN2GS1S2;
                fallback.effectiveDeterministic = deterministicRequested;
                actual = LegacyPolicyV1::SelectFallbackAfterRejection(actual, verified.rejected, fallback);
            } else {
                actual.selected = verified.plan;
            }
            variantMismatchCount += expected.variantId != actual.selected.variantId;
            deterministicMismatchCount += expected.effectiveDeterministic != actual.selected.effectiveDeterministic;
        }
    }
    EXPECT_EQ(0U, variantMismatchCount);
    EXPECT_EQ(0U, deterministicMismatchCount);
}

TEST(FagLegacyPolicyV1, InitialVariantSelectionDoesNotDependOnCandidateRegistrationOrder)
{
    const InitialVariantLimits limits = MakeInitialVariantLimits();
    InitialVariantCandidateInput input = MakeInitialVariantInput();
    const CandidateOutcome bn2 = Bn2Strategy::TryBuild(input, limits);
    const CandidateOutcome multiBlock = Bn2MultiBlockStrategy::TryBuild(input, limits);
    const CandidateOutcome bn2s2 = Bn2S2Strategy::TryBuild(input, limits);
    const CandidateOutcome fallback = Bn2Gs1S2Strategy::TryBuild(input, limits);

    InitialVariantCandidateSet forward;
    forward.Add(bn2);
    forward.Add(multiBlock);
    forward.Add(bn2s2);
    forward.Add(fallback);
    InitialVariantCandidateSet reverse;
    reverse.Add(fallback);
    reverse.Add(bn2s2);
    reverse.Add(multiBlock);
    reverse.Add(bn2);

    const SelectionResult forwardResult = LegacyPolicyV1::SelectInitialVariant(forward);
    const SelectionResult reverseResult = LegacyPolicyV1::SelectInitialVariant(reverse);
    EXPECT_EQ(static_cast<uint32_t>(VariantId::BN2), static_cast<uint32_t>(forwardResult.selected.variantId));
    EXPECT_EQ(static_cast<uint32_t>(forwardResult.selected.variantId),
              static_cast<uint32_t>(reverseResult.selected.variantId));
    EXPECT_EQ(forwardResult.winningRule, reverseResult.winningRule);
    ASSERT_EQ(forwardResult.GetRejectionCount(), reverseResult.GetRejectionCount());
    ASSERT_EQ(2U, forwardResult.GetRejectionCount());
    for (size_t index = 0; index < forwardResult.GetRejectionCount(); ++index) {
        EXPECT_EQ(static_cast<uint32_t>(forwardResult.GetRejection(index).variantId),
                  static_cast<uint32_t>(reverseResult.GetRejection(index).variantId));
        EXPECT_EQ(static_cast<uint32_t>(forwardResult.GetRejection(index).reason),
                  static_cast<uint32_t>(reverseResult.GetRejection(index).reason));
    }
}

TEST(FagLegacyPolicyV1, InitialCandidatePlannerMatchesFrozenLegacyRoute)
{
    const InitialVariantLimits limits = MakeInitialVariantLimits();
    const int64_t sequencePairs[][2] = {
        {64, 64}, {128, 128}, {129, 129}, {256, 256}, {640, 640}, {641, 641}, {64, 192}, {64, 1024}, {64, 1025},
    };
    const int64_t headDims[] = {64, 128, 129, 512, 513};
    const int64_t batchSizes[] = {1, 2, 4};
    size_t variantMismatchCount = 0;
    size_t layoutMismatchCount = 0;
    size_t deterministicMismatchCount = 0;
    for (const auto &sequencePair : sequencePairs) {
        for (const int64_t headDim : headDims) {
            for (const int64_t batchSize : batchSizes) {
                for (uint32_t mask = 0; mask < 1024; ++mask) {
                    InitialVariantCandidateInput input = MakeInitialVariantInput();
                    input.s1 = sequencePair[0];
                    input.s2 = sequencePair[1];
                    input.d = headDim;
                    input.d1 = (mask & 1U) == 0 ? headDim : 64;
                    input.b = batchSize;
                    input.executionLayout = (mask & 2U) == 0 ? 1U : limits.tndExecutionLayout;
                    input.sparseMode = (mask & 4U) == 0 ? 0U : limits.prefixSparseMode;
                    input.deterministicRequested = (mask & 8U) != 0;
                    input.queryTypeSupported = (mask & 16U) == 0;
                    input.hasRope = (mask & 32U) != 0;
                    input.dropMaskOuter = (mask & 64U) != 0;
                    input.sequenceContainsZero = (mask & 128U) != 0;
                    input.isSparse = (mask & 256U) != 0;
                    input.allSequencesSame = (mask & 512U) != 0;
                    input.dropoutDisabled = (mask & 1U) == 0;
                    input.dropoutEnabled = !input.dropoutDisabled;

                    const LegacyInitialVariantDecision expected = SelectWithFrozenLegacyExpression(input, limits);
                    const InitialVariantCandidateSet candidates = InitialVariantCandidatePlanner::Build(input, limits);
                    const SelectionResult actual = LegacyPolicyV1::SelectInitialVariant(candidates);
                    variantMismatchCount += expected.variantId != actual.selected.variantId;
                    layoutMismatchCount += expected.executionLayout != actual.selected.executionLayout;
                    deterministicMismatchCount +=
                        expected.effectiveDeterministic != actual.selected.effectiveDeterministic;
                }
            }
        }
    }
    EXPECT_EQ(0U, variantMismatchCount);
    EXPECT_EQ(0U, layoutMismatchCount);
    EXPECT_EQ(0U, deterministicMismatchCount);
}

TEST(FagRouteModel, RouteTracePreservesDecisionOrderAndReportsOverflow)
{
    RouteTrace trace;
    trace.Reset(123U, FAG_ARCH35_LEGACY_POLICY_VERSION);
    for (size_t index = 0; index < FAG_ROUTE_TRACE_CAPACITY + 3U; ++index) {
        trace.Record(RouteStage::INITIAL_VARIANT, RouteField::SPLIT_AXIS, RouteReason::INITIAL_BN2GS1S2,
                     static_cast<int64_t>(index), static_cast<int64_t>(index + 1U));
    }

    EXPECT_EQ(FAG_ROUTE_TRACE_CAPACITY, trace.GetDecisionCount());
    EXPECT_EQ(3U, trace.GetDroppedDecisionCount());
    EXPECT_EQ(0, trace.GetDecision(0).before);
    EXPECT_EQ(static_cast<int64_t>(FAG_ROUTE_TRACE_CAPACITY), trace.GetDecision(FAG_ROUTE_TRACE_CAPACITY - 1U).after);
}

TEST(FagRouteModel, SelectionResultReportsRejectedCandidateOverflow)
{
    SelectionResult result;
    CandidateRejected rejected;
    rejected.variantId = VariantId::BN2;
    for (size_t index = 0; index < FAG_CANDIDATE_REJECTION_CAPACITY + 2U; ++index) {
        rejected.actual = static_cast<int64_t>(index);
        result.RecordRejection(rejected);
    }

    EXPECT_EQ(FAG_CANDIDATE_REJECTION_CAPACITY, result.GetRejectionCount());
    EXPECT_EQ(2U, result.GetDroppedRejectionCount());
    EXPECT_EQ(0, result.GetRejection(0).actual);
    EXPECT_EQ(static_cast<int64_t>(FAG_CANDIDATE_REJECTION_CAPACITY - 1U),
              result.GetRejection(FAG_CANDIDATE_REJECTION_CAPACITY - 1U).actual);
}

TEST(FagRouteModel, RouteTraceFinalizesSelectedVariantAndKey)
{
    RouteTrace trace;
    trace.Reset(456U, FAG_ARCH35_LEGACY_POLICY_VERSION);
    trace.Finalize(VariantId::BN2S2, 789U);

    EXPECT_EQ(static_cast<uint32_t>(VariantId::BN2S2), static_cast<uint32_t>(trace.GetSelectedVariant()));
    EXPECT_EQ(789U, trace.GetFinalTilingKey());
    ASSERT_EQ(1U, trace.GetDecisionCount());
    EXPECT_EQ(static_cast<uint32_t>(RouteReason::FINAL_KEY_ENCODED),
              static_cast<uint32_t>(trace.GetDecision(0).reason));
}

TEST(FagLegacyPolicyV1, ResolvesCompositeVariantsWithoutLeakingBooleanCombinations)
{
    EXPECT_EQ(static_cast<uint32_t>(VariantId::BN2_MULTI_BLOCK),
              static_cast<uint32_t>(LegacyPolicyV1::ResolveVariant(true, true, false)));
    EXPECT_EQ(static_cast<uint32_t>(VariantId::BN2),
              static_cast<uint32_t>(LegacyPolicyV1::ResolveVariant(true, false, false)));
    EXPECT_EQ(static_cast<uint32_t>(VariantId::BN2S2),
              static_cast<uint32_t>(LegacyPolicyV1::ResolveVariant(false, false, true)));
    EXPECT_EQ(static_cast<uint32_t>(VariantId::BN2GS1S2),
              static_cast<uint32_t>(LegacyPolicyV1::ResolveVariant(false, false, false)));
    EXPECT_EQ(static_cast<uint32_t>(VariantId::BN2GS1S2),
              static_cast<uint32_t>(LegacyPolicyV1::ResolveVariant(false, true, false)));
}

TEST(FagKernelCapability, NzOutputRequiresACompiledVariantAndUnalignedInteriorHeadDim)
{
    NzOutputCapabilityInput input;
    input.isBn2Gs1S2 = true;
    input.queryTypeSupported = true;
    input.isLegacyDeterministic = false;
    input.minimumHeadDimExclusive = 64;
    input.maximumHeadDimExclusive = 128;
    input.headDimAlignment = 16;

    input.headDim = 64;
    EXPECT_FALSE(KernelCapabilityVerifier::SupportsNzOutput(input));
    input.headDim = 65;
    EXPECT_TRUE(KernelCapabilityVerifier::SupportsNzOutput(input));
    input.headDim = 80;
    EXPECT_FALSE(KernelCapabilityVerifier::SupportsNzOutput(input));
    input.headDim = 127;
    EXPECT_TRUE(KernelCapabilityVerifier::SupportsNzOutput(input));
    input.headDim = 128;
    EXPECT_FALSE(KernelCapabilityVerifier::SupportsNzOutput(input));

    input.headDim = 65;
    input.isBn2Gs1S2 = false;
    EXPECT_FALSE(KernelCapabilityVerifier::SupportsNzOutput(input));
    input.isBn2Gs1S2 = true;
    input.queryTypeSupported = false;
    EXPECT_FALSE(KernelCapabilityVerifier::SupportsNzOutput(input));
    input.queryTypeSupported = true;
    input.isLegacyDeterministic = true;
    EXPECT_FALSE(KernelCapabilityVerifier::SupportsNzOutput(input));
}

TEST(FagLegacyPolicyV1, SeparatesNzCapabilityFromPerformancePolicy)
{
    LegacyOutputFormatPolicyInput input;
    input.nzKernelSupported = false;
    input.enableSwizzle = true;
    input.s1 = 4096;
    input.s2 = 4096;
    input.minimumSequenceLength = 2048;

    OutputFormatDecision decision = LegacyPolicyV1::SelectOutputFormat(input);
    EXPECT_FALSE(decision.capabilityAccepted);
    EXPECT_TRUE(decision.performanceAccepted);
    EXPECT_EQ(static_cast<uint32_t>(OutputFormat::ND), static_cast<uint32_t>(decision.plan.format));

    input.nzKernelSupported = true;
    decision = LegacyPolicyV1::SelectOutputFormat(input);
    EXPECT_TRUE(decision.capabilityAccepted);
    EXPECT_TRUE(decision.performanceAccepted);
    EXPECT_EQ(static_cast<uint32_t>(OutputFormat::NZ), static_cast<uint32_t>(decision.plan.format));
    EXPECT_EQ(FAG_RULE_LEGACY_NZ_LARGE_SEQUENCE, decision.winningRule);
}

TEST(FagLegacyPolicyV1, NzSequenceThresholdHasExplicitBoundaries)
{
    LegacyOutputFormatPolicyInput input;
    input.nzKernelSupported = true;
    input.enableSwizzle = true;
    input.minimumSequenceLength = 2048;
    input.s2 = 2048;

    input.s1 = 2047;
    OutputFormatDecision decision = LegacyPolicyV1::SelectOutputFormat(input);
    EXPECT_FALSE(decision.performanceAccepted);
    EXPECT_EQ(static_cast<uint32_t>(OutputFormat::ND), static_cast<uint32_t>(decision.plan.format));

    input.s1 = 2048;
    decision = LegacyPolicyV1::SelectOutputFormat(input);
    EXPECT_TRUE(decision.performanceAccepted);
    EXPECT_EQ(static_cast<uint32_t>(OutputFormat::NZ), static_cast<uint32_t>(decision.plan.format));

    input.s1 = 2049;
    decision = LegacyPolicyV1::SelectOutputFormat(input);
    EXPECT_TRUE(decision.performanceAccepted);
    EXPECT_EQ(static_cast<uint32_t>(OutputFormat::NZ), static_cast<uint32_t>(decision.plan.format));
}

} // namespace
} // namespace fag
} // namespace optiling
