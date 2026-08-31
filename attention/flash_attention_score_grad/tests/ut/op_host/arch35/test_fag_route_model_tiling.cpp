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
#include "../../../../op_host/arch35/model/fag_route_model.h"
#include "../../../../op_host/arch35/policy/fag_legacy_policy_v1.h"

namespace optiling {
namespace fag {
namespace {

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
