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

#include <array>
#include <cstddef>
#include <cstdint>

namespace optiling {
namespace fag {

using PolicyVersion = uint32_t;
constexpr PolicyVersion FAG_ARCH35_LEGACY_POLICY_VERSION = 1U;
constexpr size_t FAG_ROUTE_TRACE_CAPACITY = 32U;
constexpr size_t FAG_INITIAL_VARIANT_CANDIDATE_CAPACITY = 4U;
constexpr size_t FAG_CANDIDATE_REJECTION_CAPACITY = 8U;
constexpr size_t FAG_CORE_PARTITION_RANGE_CAPACITY = 36U;

using RuleId = uint32_t;
constexpr RuleId FAG_RULE_LEGACY_DEFAULT = 1000U;
constexpr RuleId FAG_RULE_LEGACY_NZ_LARGE_SEQUENCE = 1001U;
constexpr RuleId FAG_RULE_LEGACY_INITIAL_BN2_MULTI_BLOCK = 1100U;
constexpr RuleId FAG_RULE_LEGACY_INITIAL_BN2 = 1101U;
constexpr RuleId FAG_RULE_LEGACY_INITIAL_BN2S2 = 1102U;
constexpr RuleId FAG_RULE_LEGACY_INITIAL_BN2GS1S2 = 1103U;

enum class VariantId : uint8_t {
    UNKNOWN = 0,
    BN2 = 1,
    BN2_MULTI_BLOCK = 2,
    BN2S2 = 3,
    BN2GS1S2 = 4,
};

enum class SourceLayoutId : uint8_t {
    UNKNOWN = 0,
    SBH = 1,
    BSH = 2,
    BNSD = 3,
    TND = 4,
    BSND = 5,
};

enum class OutputFormat : uint8_t {
    ND = 0,
    NZ,
};

enum class CandidateField : uint8_t {
    NONE = 0,
    S1 = 1,
    S2 = 2,
    BATCH_HEAD_PRODUCT = 3,
    HEAD_TOPOLOGY = 4,
    HEAD_DIM = 5,
    VALUE_HEAD_DIM = 6,
    QUERY_TYPE = 7,
    SEQUENCE_ZERO = 8,
    DROP_MASK = 9,
    EXECUTION_LAYOUT = 10,
    SPARSE_MODE = 11,
    ROPE = 12,
    DETERMINISTIC = 13,
    DROPOUT = 14,
    CORE_PARTITION = 15,
    ACTIVE_CORE_COUNT = 16,
    INVALID_BASIC_BLOCK = 17,
    CORE_RANGE_CAPACITY = 18,
};

enum class CandidateRejectStage : uint8_t {
    COARSE_CAPABILITY = 0,
    CORE_PARTITION = 1,
    SPARSE_PLANNING = 2,
};

enum class CandidateRejectReason : uint16_t {
    NONE = 0,
    SEQUENCE_RANGE_UNSUPPORTED = 1000,
    HEAD_TOPOLOGY_UNSUPPORTED = 1001,
    HEAD_DIM_UNSUPPORTED = 1002,
    QUERY_TYPE_UNSUPPORTED = 1003,
    SEQUENCE_CONTAINS_ZERO = 1004,
    DROP_MASK_UNSUPPORTED = 1005,
    BATCH_HEAD_TOO_SMALL = 1006,
    EXECUTION_LAYOUT_UNSUPPORTED = 1007,
    SPARSE_MODE_UNSUPPORTED = 1008,
    ROPE_UNSUPPORTED = 1009,
    VALUE_HEAD_DIM_MISMATCH = 1010,
    SPARSE_OR_DETERMINISTIC_UNSUPPORTED = 1011,
    DROPOUT_HEAD_DIM_UNSUPPORTED = 1012,
    CORE_PARTITION_UNAVAILABLE = 1013,
    INSUFFICIENT_CORE_UTILIZATION = 1014,
    INVALID_BASIC_BLOCK = 1015,
    CORE_RANGE_CAPACITY_EXCEEDED = 1016,
};

enum class RouteStage : uint8_t {
    VALIDATED_PROBLEM = 0,
    INITIAL_VARIANT = 1,
    DETERMINISTIC_PLANNING = 2,
    SPARSE_PLANNING = 3,
    CORE_PARTITION = 4,
    SCHEDULE = 5,
    OUTPUT_FORMAT = 6,
    WORKSPACE = 7,
    ENCODER = 8,
};

enum class RouteField : uint8_t {
    SOURCE_LAYOUT = 0,
    EXECUTION_LAYOUT = 1,
    DETERMINISTIC_REQUESTED = 2,
    EFFECTIVE_DETERMINISTIC = 3,
    SPLIT_AXIS = 4,
    BLOCK_OUTER = 5,
    ENABLE_SWIZZLE = 6,
    TND_SWIZZLE = 7,
    OUTPUT_FORMAT = 8,
    BLOCK_SCHEDULE = 9,
    WORKSPACE_BYTES = 10,
    BLOCK_DIM = 11,
    TILING_KEY = 12,
};

enum class RouteReason : uint16_t {
    INPUT_FACT_CAPTURED = 0,
    INITIAL_BN2 = 100,
    INITIAL_BN2_MULTI_BLOCK = 101,
    INITIAL_BN2S2 = 102,
    INITIAL_BN2GS1S2 = 103,
    BN2_DISABLES_DETERMINISTIC = 104,
    BN2S2_EXECUTION_LAYOUT = 105,
    TND_EQUAL_LENGTH_EXECUTION_LAYOUT = 106,
    BN2S2_INSUFFICIENT_CORES = 200,
    BN2_MULTI_BLOCK_INVALID_BLOCK = 201,
    LEGACY_SPARSE_NORMALIZATION = 202,
    TND_DETER_CAUSAL_REQUIRES_BN2GS1S2 = 203,
    TND_DETER_BN2S2_UNSUPPORTED = 204,
    TND_DETER_SYNC_COST_FALLBACK = 205,
    CORE_PARTITION_SELECTED = 300,
    L2_OR_INVALID_BLOCK_SWIZZLE = 400,
    TND_SWIZZLE_SELECTED = 401,
    LEGACY_NZ_CAPABILITY_REJECTED = 500,
    LEGACY_NZ_POLICY_REJECTED = 501,
    LEGACY_NZ_POLICY_SELECTED = 502,
    BLOCK_SCHEDULE_SELECTED = 600,
    WORKSPACE_PLANNED = 700,
    BLOCK_DIM_SELECTED = 800,
    FINAL_KEY_ENCODED = 801,
};

inline const char *GetRouteReasonName(RouteReason reason)
{
    switch (reason) {
        case RouteReason::INPUT_FACT_CAPTURED:
            return "input_fact_captured";
        case RouteReason::INITIAL_BN2:
            return "initial_bn2";
        case RouteReason::INITIAL_BN2_MULTI_BLOCK:
            return "initial_bn2_multi_block";
        case RouteReason::INITIAL_BN2S2:
            return "initial_bn2s2";
        case RouteReason::INITIAL_BN2GS1S2:
            return "initial_bn2gs1s2";
        case RouteReason::BN2_DISABLES_DETERMINISTIC:
            return "bn2_disables_deterministic";
        case RouteReason::BN2S2_EXECUTION_LAYOUT:
            return "bn2s2_execution_layout";
        case RouteReason::TND_EQUAL_LENGTH_EXECUTION_LAYOUT:
            return "tnd_equal_length_execution_layout";
        case RouteReason::BN2S2_INSUFFICIENT_CORES:
            return "bn2s2_insufficient_cores";
        case RouteReason::BN2_MULTI_BLOCK_INVALID_BLOCK:
            return "bn2_multi_block_invalid_block";
        case RouteReason::LEGACY_SPARSE_NORMALIZATION:
            return "legacy_sparse_normalization";
        case RouteReason::TND_DETER_CAUSAL_REQUIRES_BN2GS1S2:
            return "tnd_deter_causal_requires_bn2gs1s2";
        case RouteReason::TND_DETER_BN2S2_UNSUPPORTED:
            return "tnd_deter_bn2s2_unsupported";
        case RouteReason::TND_DETER_SYNC_COST_FALLBACK:
            return "tnd_deter_sync_cost_fallback";
        case RouteReason::CORE_PARTITION_SELECTED:
            return "core_partition_selected";
        case RouteReason::L2_OR_INVALID_BLOCK_SWIZZLE:
            return "l2_or_invalid_block_swizzle";
        case RouteReason::TND_SWIZZLE_SELECTED:
            return "tnd_swizzle_selected";
        case RouteReason::LEGACY_NZ_CAPABILITY_REJECTED:
            return "legacy_nz_capability_rejected";
        case RouteReason::LEGACY_NZ_POLICY_REJECTED:
            return "legacy_nz_policy_rejected";
        case RouteReason::LEGACY_NZ_POLICY_SELECTED:
            return "legacy_nz_policy_selected";
        case RouteReason::BLOCK_SCHEDULE_SELECTED:
            return "block_schedule_selected";
        case RouteReason::WORKSPACE_PLANNED:
            return "workspace_planned";
        case RouteReason::BLOCK_DIM_SELECTED:
            return "block_dim_selected";
        case RouteReason::FINAL_KEY_ENCODED:
            return "final_key_encoded";
        default:
            return "unknown";
    }
}

struct ProblemFacts {
    SourceLayoutId sourceLayout = SourceLayoutId::UNKNOWN;
    uint32_t queryType = 0;
    uint32_t sparseMode = 0;
    uint32_t pseOptional = 0;
    uint32_t attenMaskOptional = 0;
    uint32_t sinkOptional = 0;
    uint32_t outputType = 0;
    uint32_t keepProbBits = 0;
    bool deterministicRequested = false;
    bool hasRope = false;
    bool dropMaskOuter = false;
    bool allSequencesSame = false;
    int64_t b = 0;
    int64_t s1 = 0;
    int64_t s2 = 0;
    int64_t n1 = 0;
    int64_t n2 = 0;
    int64_t g = 0;
    int64_t d = 0;
    int64_t d1 = 0;
    uint64_t aicNum = 0;
    uint64_t aivNum = 0;
    uint64_t l2CacheSize = 0;
    uint64_t actualSequenceFingerprint = 0;
    uint64_t tailZeroCount = 0;

    uint64_t Fingerprint() const
    {
        uint64_t hash = 1469598103934665603ULL;
        HashValue(hash, sourceLayout);
        HashValue(hash, queryType);
        HashValue(hash, sparseMode);
        HashValue(hash, pseOptional);
        HashValue(hash, attenMaskOptional);
        HashValue(hash, sinkOptional);
        HashValue(hash, outputType);
        HashValue(hash, keepProbBits);
        HashValue(hash, deterministicRequested);
        HashValue(hash, hasRope);
        HashValue(hash, dropMaskOuter);
        HashValue(hash, allSequencesSame);
        HashValue(hash, b);
        HashValue(hash, s1);
        HashValue(hash, s2);
        HashValue(hash, n1);
        HashValue(hash, n2);
        HashValue(hash, g);
        HashValue(hash, d);
        HashValue(hash, d1);
        HashValue(hash, aicNum);
        HashValue(hash, aivNum);
        HashValue(hash, l2CacheSize);
        HashValue(hash, actualSequenceFingerprint);
        HashValue(hash, tailZeroCount);
        return hash;
    }

private:
    template <typename T>
    static void HashValue(uint64_t &hash, T value)
    {
        uint64_t data = static_cast<uint64_t>(value);
        for (size_t byteIndex = 0; byteIndex < sizeof(data); ++byteIndex) {
            hash ^= data & 0xFFULL;
            hash *= 1099511628211ULL;
            data >>= 8U;
        }
    }
};

struct SplitAxisPlan {
    uint32_t axis = 0;
    bool isMultiBlock = false;
};

struct TilePlan {
    uint32_t s1Inner = 0;
    uint32_t s2Inner = 0;
    uint32_t dInner = 0;
};

enum class CorePartitionMode : uint8_t {
    UNKNOWN = 0,
    FUSED_BN2GS1S2 = 1,
    FUSED_BN2GS2 = 2,
    FUSED_BN2G = 3,
};

struct WorkRange {
    int64_t begin = 0;
    int64_t end = 0;
};

struct CorePartitionPlan {
    CorePartitionMode mode = CorePartitionMode::UNKNOWN;
    uint32_t activeCoreCount = 0;
    int64_t blockFactor = 0;
    int64_t totalWork = 0;
    uint64_t maxCoreWork = 0;
    bool isSplitByBlockIndex = false;
    std::array<WorkRange, FAG_CORE_PARTITION_RANGE_CAPACITY> ranges{};
};

struct SparsePlan {
    uint32_t sparseType = 0;
    uint32_t deterministicSparseType = 0;
};

struct OutputFormatPlan {
    OutputFormat format = OutputFormat::ND;
    uint64_t alignedDqBytes = 0;
    uint64_t alignedDkBytes = 0;
    uint64_t alignedDvBytes = 0;
    uint64_t formatConvertBytes = 0;
};

struct WorkspacePlan {
    uint64_t workspaceBytes = 0;
};

struct RouteMetrics {
    uint32_t activeAic = 0;
    uint32_t activeAiv = 0;
    uint64_t maxCoreWork = 0;
    uint64_t averageCoreWork = 0;
    uint64_t invalidSlotCount = 0;
    uint64_t syncRoundCount = 0;
    uint64_t atomicConflictCount = 0;
    uint64_t estimatedL2Footprint = 0;
    uint64_t workspaceBytes = 0;
    uint64_t formatConvertBytes = 0;
};

struct CandidatePlan {
    VariantId variantId = VariantId::UNKNOWN;
    SplitAxisPlan split;
    TilePlan tile;
    CorePartitionPlan cores;
    SparsePlan sparse;
    OutputFormatPlan outputFormat;
    WorkspacePlan workspace;
    RouteMetrics metrics;
    uint32_t executionLayout = 0;
    bool effectiveDeterministic = false;
    uint32_t blockDim = 0;
    uint64_t tilingKey = 0;
};

struct CandidateRejected {
    VariantId variantId = VariantId::UNKNOWN;
    CandidateRejectStage stage = CandidateRejectStage::COARSE_CAPABILITY;
    CandidateRejectReason reason = CandidateRejectReason::NONE;
    CandidateField field = CandidateField::NONE;
    int64_t actual = 0;
    int64_t expected = 0;
};

struct CandidateOutcome {
    bool accepted = false;
    CandidatePlan plan;
    CandidateRejected rejected;
};

struct CorePartitionOutcome {
    bool accepted = false;
    CorePartitionPlan plan;
    CandidateRejected rejected;
};

struct SelectionResult {
    CandidatePlan selected;
    PolicyVersion policyVersion = FAG_ARCH35_LEGACY_POLICY_VERSION;
    RuleId winningRule = FAG_RULE_LEGACY_DEFAULT;

    void RecordRejection(const CandidateRejected &rejected)
    {
        if (rejectionCount_ < rejections_.size()) {
            rejections_[rejectionCount_++] = rejected;
        } else {
            ++droppedRejectionCount_;
        }
    }

    size_t GetRejectionCount() const
    {
        return rejectionCount_;
    }

    const CandidateRejected &GetRejection(size_t index) const
    {
        return rejections_[index];
    }

    size_t GetDroppedRejectionCount() const
    {
        return droppedRejectionCount_;
    }

private:
    std::array<CandidateRejected, FAG_CANDIDATE_REJECTION_CAPACITY> rejections_;
    size_t rejectionCount_ = 0;
    size_t droppedRejectionCount_ = 0;
};

struct RouteDecision {
    RouteStage stage = RouteStage::VALIDATED_PROBLEM;
    RouteField field = RouteField::SOURCE_LAYOUT;
    RouteReason reason = RouteReason::INPUT_FACT_CAPTURED;
    int64_t before = 0;
    int64_t after = 0;
};

class RouteTrace {
public:
    void Reset(uint64_t inputFingerprint, PolicyVersion policyVersion)
    {
        inputFingerprint_ = inputFingerprint;
        policyVersion_ = policyVersion;
        selectedVariant_ = VariantId::UNKNOWN;
        finalTilingKey_ = 0;
        decisionCount_ = 0;
        droppedDecisionCount_ = 0;
    }

    void Record(RouteStage stage, RouteField field, RouteReason reason, int64_t before, int64_t after)
    {
        if (decisionCount_ >= decisions_.size()) {
            ++droppedDecisionCount_;
            return;
        }
        RouteDecision &decision = decisions_[decisionCount_++];
        decision.stage = stage;
        decision.field = field;
        decision.reason = reason;
        decision.before = before;
        decision.after = after;
    }

    void Finalize(VariantId selectedVariant, uint64_t tilingKey)
    {
        selectedVariant_ = selectedVariant;
        finalTilingKey_ = tilingKey;
        Record(RouteStage::ENCODER, RouteField::TILING_KEY, RouteReason::FINAL_KEY_ENCODED, 0,
               static_cast<int64_t>(tilingKey));
    }

    uint64_t GetInputFingerprint() const
    {
        return inputFingerprint_;
    }

    PolicyVersion GetPolicyVersion() const
    {
        return policyVersion_;
    }

    VariantId GetSelectedVariant() const
    {
        return selectedVariant_;
    }

    uint64_t GetFinalTilingKey() const
    {
        return finalTilingKey_;
    }

    size_t GetDecisionCount() const
    {
        return decisionCount_;
    }

    size_t GetDroppedDecisionCount() const
    {
        return droppedDecisionCount_;
    }

    const RouteDecision &GetDecision(size_t index) const
    {
        return decisions_[index];
    }

private:
    uint64_t inputFingerprint_ = 0;
    PolicyVersion policyVersion_ = FAG_ARCH35_LEGACY_POLICY_VERSION;
    VariantId selectedVariant_ = VariantId::UNKNOWN;
    uint64_t finalTilingKey_ = 0;
    std::array<RouteDecision, FAG_ROUTE_TRACE_CAPACITY> decisions_;
    size_t decisionCount_ = 0;
    size_t droppedDecisionCount_ = 0;
};

} // namespace fag
} // namespace optiling
