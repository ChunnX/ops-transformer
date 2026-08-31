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

struct InitialVariantCandidateInput {
    int64_t b = 0;
    int64_t s1 = 0;
    int64_t s2 = 0;
    int64_t n1 = 0;
    int64_t n2 = 0;
    int64_t d = 0;
    int64_t d1 = 0;
    uint64_t tailZeroCount = 0;
    uint32_t executionLayout = 0;
    uint32_t sparseMode = 0;
    bool queryTypeSupported = false;
    bool sequenceContainsZero = false;
    bool isSparse = false;
    bool deterministicRequested = false;
    bool hasRope = false;
    bool dropMaskOuter = false;
    bool allSequencesSame = false;
    bool dropoutDisabled = false;
    bool dropoutEnabled = false;
};

struct InitialVariantLimits {
    int64_t bn2MaxSequence = 0;
    int64_t bn2MaxHeadDim = 0;
    int64_t bn2MultiBlockMaxSequence = 0;
    int64_t bn2MultiBlockBatchHeadLow = 0;
    int64_t bn2MultiBlockBatchHeadHigh = 0;
    int64_t bn2s2MaxSequence = 0;
    int64_t bn2s2WriteUbMaxHeadDim = 0;
    int64_t bn2TndMaxHeadDim = 0;
    int64_t bn2MultiBlockSequenceAlignment = 0;
    int64_t bn2s2MinimumSequenceGap = 0;
    int64_t bn2s2DropoutMaxHeadDim = 0;
    uint32_t tndExecutionLayout = 0;
    uint32_t prefixSparseMode = 0;
    uint32_t prefixCompressSparseMode = 0;
    uint32_t bn2SplitAxis = 0;
    uint32_t bn2s2SplitAxis = 0;
    uint32_t bn2gs1s2SplitAxis = 0;
};

class InitialVariantCandidateSet {
public:
    void Add(const CandidateOutcome &outcome)
    {
        if (size_ < outcomes_.size()) {
            outcomes_[size_++] = outcome;
        }
    }

    const CandidateOutcome *Find(VariantId variantId) const
    {
        for (size_t index = 0; index < size_; ++index) {
            const CandidateOutcome &outcome = outcomes_[index];
            const VariantId current = outcome.accepted ? outcome.plan.variantId : outcome.rejected.variantId;
            if (current == variantId) {
                return &outcome;
            }
        }
        return nullptr;
    }

    size_t Size() const
    {
        return size_;
    }

    const CandidateOutcome &Get(size_t index) const
    {
        return outcomes_[index];
    }

private:
    std::array<CandidateOutcome, FAG_INITIAL_VARIANT_CANDIDATE_CAPACITY> outcomes_;
    size_t size_ = 0;
};

namespace candidate_detail {

inline CandidateOutcome Accept(VariantId variantId, uint32_t splitAxis, bool isMultiBlock, uint32_t executionLayout,
                               bool effectiveDeterministic)
{
    CandidateOutcome outcome;
    outcome.accepted = true;
    outcome.plan.variantId = variantId;
    outcome.plan.split.axis = splitAxis;
    outcome.plan.split.isMultiBlock = isMultiBlock;
    outcome.plan.executionLayout = executionLayout;
    outcome.plan.effectiveDeterministic = effectiveDeterministic;
    return outcome;
}

inline CandidateOutcome Reject(VariantId variantId, CandidateRejectReason reason, CandidateField field, int64_t actual,
                               int64_t expected)
{
    CandidateOutcome outcome;
    outcome.rejected.variantId = variantId;
    outcome.rejected.reason = reason;
    outcome.rejected.field = field;
    outcome.rejected.actual = actual;
    outcome.rejected.expected = expected;
    return outcome;
}

} // namespace candidate_detail

class Bn2Strategy {
public:
    static CandidateOutcome TryBuild(const InitialVariantCandidateInput &input, const InitialVariantLimits &limits)
    {
        if (input.s1 > limits.bn2MaxSequence) {
            return candidate_detail::Reject(VariantId::BN2, CandidateRejectReason::SEQUENCE_RANGE_UNSUPPORTED,
                                            CandidateField::S1, input.s1, limits.bn2MaxSequence);
        }
        if (input.s2 > limits.bn2MaxSequence) {
            return candidate_detail::Reject(VariantId::BN2, CandidateRejectReason::SEQUENCE_RANGE_UNSUPPORTED,
                                            CandidateField::S2, input.s2, limits.bn2MaxSequence);
        }
        if (input.n1 != input.n2) {
            return candidate_detail::Reject(VariantId::BN2, CandidateRejectReason::HEAD_TOPOLOGY_UNSUPPORTED,
                                            CandidateField::HEAD_TOPOLOGY, input.n1, input.n2);
        }
        if (input.d > limits.bn2MaxHeadDim) {
            return candidate_detail::Reject(VariantId::BN2, CandidateRejectReason::HEAD_DIM_UNSUPPORTED,
                                            CandidateField::HEAD_DIM, input.d, limits.bn2MaxHeadDim);
        }
        if (!input.queryTypeSupported) {
            return candidate_detail::Reject(VariantId::BN2, CandidateRejectReason::QUERY_TYPE_UNSUPPORTED,
                                            CandidateField::QUERY_TYPE, 0, 1);
        }
        if (input.tailZeroCount != 0 || input.sequenceContainsZero) {
            return candidate_detail::Reject(VariantId::BN2, CandidateRejectReason::SEQUENCE_CONTAINS_ZERO,
                                            CandidateField::SEQUENCE_ZERO, 1, 0);
        }
        if (input.executionLayout == limits.tndExecutionLayout && input.d > limits.bn2TndMaxHeadDim) {
            return candidate_detail::Reject(VariantId::BN2, CandidateRejectReason::EXECUTION_LAYOUT_UNSUPPORTED,
                                            CandidateField::HEAD_DIM, input.d, limits.bn2TndMaxHeadDim);
        }
        if (input.dropMaskOuter) {
            return candidate_detail::Reject(VariantId::BN2, CandidateRejectReason::DROP_MASK_UNSUPPORTED,
                                            CandidateField::DROP_MASK, 1, 0);
        }
        return candidate_detail::Accept(VariantId::BN2, limits.bn2SplitAxis, false, input.executionLayout, false);
    }
};

class Bn2MultiBlockStrategy {
public:
    static CandidateOutcome TryBuild(const InitialVariantCandidateInput &input, const InitialVariantLimits &limits)
    {
        const int64_t batchHeadProduct = input.b * input.n1;
        const bool alignedLowBatchHead = batchHeadProduct >= limits.bn2MultiBlockBatchHeadLow &&
                                         limits.bn2MultiBlockSequenceAlignment > 0 &&
                                         input.s1 % limits.bn2MultiBlockSequenceAlignment == 0 &&
                                         input.s2 % limits.bn2MultiBlockSequenceAlignment == 0;
        if (batchHeadProduct < limits.bn2MultiBlockBatchHeadHigh && !alignedLowBatchHead) {
            return candidate_detail::Reject(VariantId::BN2_MULTI_BLOCK, CandidateRejectReason::BATCH_HEAD_TOO_SMALL,
                                            CandidateField::BATCH_HEAD_PRODUCT, batchHeadProduct,
                                            limits.bn2MultiBlockBatchHeadLow);
        }
        if (input.executionLayout == limits.tndExecutionLayout) {
            return candidate_detail::Reject(
                VariantId::BN2_MULTI_BLOCK, CandidateRejectReason::EXECUTION_LAYOUT_UNSUPPORTED,
                CandidateField::EXECUTION_LAYOUT, input.executionLayout, limits.tndExecutionLayout);
        }
        if (input.sparseMode == limits.prefixSparseMode || input.sparseMode == limits.prefixCompressSparseMode) {
            return candidate_detail::Reject(VariantId::BN2_MULTI_BLOCK, CandidateRejectReason::SPARSE_MODE_UNSUPPORTED,
                                            CandidateField::SPARSE_MODE, input.sparseMode, 0);
        }
        if (input.s1 <= limits.bn2MaxSequence && input.s2 <= limits.bn2MaxSequence) {
            return candidate_detail::Reject(VariantId::BN2_MULTI_BLOCK,
                                            CandidateRejectReason::SEQUENCE_RANGE_UNSUPPORTED, CandidateField::S1,
                                            input.s1, limits.bn2MaxSequence + 1);
        }
        if (input.s1 > limits.bn2MultiBlockMaxSequence) {
            return candidate_detail::Reject(VariantId::BN2_MULTI_BLOCK,
                                            CandidateRejectReason::SEQUENCE_RANGE_UNSUPPORTED, CandidateField::S1,
                                            input.s1, limits.bn2MultiBlockMaxSequence);
        }
        if (input.s2 > limits.bn2MultiBlockMaxSequence) {
            return candidate_detail::Reject(VariantId::BN2_MULTI_BLOCK,
                                            CandidateRejectReason::SEQUENCE_RANGE_UNSUPPORTED, CandidateField::S2,
                                            input.s2, limits.bn2MultiBlockMaxSequence);
        }
        if (input.n1 != input.n2) {
            return candidate_detail::Reject(VariantId::BN2_MULTI_BLOCK,
                                            CandidateRejectReason::HEAD_TOPOLOGY_UNSUPPORTED,
                                            CandidateField::HEAD_TOPOLOGY, input.n1, input.n2);
        }
        if (input.d > limits.bn2MaxHeadDim) {
            return candidate_detail::Reject(VariantId::BN2_MULTI_BLOCK, CandidateRejectReason::HEAD_DIM_UNSUPPORTED,
                                            CandidateField::HEAD_DIM, input.d, limits.bn2MaxHeadDim);
        }
        if (!input.queryTypeSupported) {
            return candidate_detail::Reject(VariantId::BN2_MULTI_BLOCK, CandidateRejectReason::QUERY_TYPE_UNSUPPORTED,
                                            CandidateField::QUERY_TYPE, 0, 1);
        }
        if (input.d != input.d1) {
            return candidate_detail::Reject(VariantId::BN2_MULTI_BLOCK, CandidateRejectReason::VALUE_HEAD_DIM_MISMATCH,
                                            CandidateField::VALUE_HEAD_DIM, input.d1, input.d);
        }
        if (input.hasRope) {
            return candidate_detail::Reject(VariantId::BN2_MULTI_BLOCK, CandidateRejectReason::ROPE_UNSUPPORTED,
                                            CandidateField::ROPE, 1, 0);
        }
        if (input.dropMaskOuter) {
            return candidate_detail::Reject(VariantId::BN2_MULTI_BLOCK, CandidateRejectReason::DROP_MASK_UNSUPPORTED,
                                            CandidateField::DROP_MASK, 1, 0);
        }
        return candidate_detail::Accept(VariantId::BN2_MULTI_BLOCK, limits.bn2SplitAxis, true, input.executionLayout,
                                        false);
    }
};

class Bn2S2Strategy {
public:
    static CandidateOutcome TryBuild(const InitialVariantCandidateInput &input, const InitialVariantLimits &limits)
    {
        if (input.hasRope) {
            return candidate_detail::Reject(VariantId::BN2S2, CandidateRejectReason::ROPE_UNSUPPORTED,
                                            CandidateField::ROPE, 1, 0);
        }
        if (input.d > limits.bn2MaxHeadDim) {
            return candidate_detail::Reject(VariantId::BN2S2, CandidateRejectReason::HEAD_DIM_UNSUPPORTED,
                                            CandidateField::HEAD_DIM, input.d, limits.bn2MaxHeadDim);
        }
        const bool nonTndRoute = input.s1 < input.s2 && input.s2 <= limits.bn2s2MaxSequence &&
                                 input.s2 - input.s1 >= limits.bn2s2MinimumSequenceGap &&
                                 input.d <= limits.bn2s2WriteUbMaxHeadDim && !input.isSparse &&
                                 !input.deterministicRequested;
        const bool layoutSupported = input.executionLayout == limits.tndExecutionLayout ||
                                     (input.allSequencesSame && !input.deterministicRequested) || nonTndRoute;
        if (!layoutSupported) {
            return candidate_detail::Reject(
                VariantId::BN2S2, CandidateRejectReason::SPARSE_OR_DETERMINISTIC_UNSUPPORTED,
                CandidateField::EXECUTION_LAYOUT, input.executionLayout, limits.tndExecutionLayout);
        }
        if (!(input.dropoutDisabled || (input.d <= limits.bn2s2DropoutMaxHeadDim && input.dropoutEnabled))) {
            return candidate_detail::Reject(VariantId::BN2S2, CandidateRejectReason::DROPOUT_HEAD_DIM_UNSUPPORTED,
                                            CandidateField::DROPOUT, input.d, limits.bn2s2DropoutMaxHeadDim);
        }
        if (input.n1 != input.n2) {
            return candidate_detail::Reject(VariantId::BN2S2, CandidateRejectReason::HEAD_TOPOLOGY_UNSUPPORTED,
                                            CandidateField::HEAD_TOPOLOGY, input.n1, input.n2);
        }
        if (!input.queryTypeSupported) {
            return candidate_detail::Reject(VariantId::BN2S2, CandidateRejectReason::QUERY_TYPE_UNSUPPORTED,
                                            CandidateField::QUERY_TYPE, 0, 1);
        }
        const uint32_t executionLayout = input.allSequencesSame ? limits.tndExecutionLayout : input.executionLayout;
        return candidate_detail::Accept(VariantId::BN2S2, limits.bn2s2SplitAxis, false, executionLayout,
                                        input.deterministicRequested);
    }
};

class Bn2Gs1S2Strategy {
public:
    static CandidateOutcome TryBuild(const InitialVariantCandidateInput &input, const InitialVariantLimits &limits)
    {
        return candidate_detail::Accept(VariantId::BN2GS1S2, limits.bn2gs1s2SplitAxis, false, input.executionLayout,
                                        input.deterministicRequested);
    }
};

class InitialVariantCandidatePlanner {
public:
    static InitialVariantCandidateSet Build(const InitialVariantCandidateInput &input,
                                            const InitialVariantLimits &limits)
    {
        InitialVariantCandidateSet candidates;
        candidates.Add(Bn2Strategy::TryBuild(input, limits));
        candidates.Add(Bn2MultiBlockStrategy::TryBuild(input, limits));
        candidates.Add(Bn2S2Strategy::TryBuild(input, limits));
        candidates.Add(Bn2Gs1S2Strategy::TryBuild(input, limits));
        return candidates;
    }
};

} // namespace fag
} // namespace optiling
