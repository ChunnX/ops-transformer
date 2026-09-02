/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

// Local, copied Ascend-C resource ledger. Keep one copy in each solution.
// Provenance: Ascend910_9382 / ascend910_93 with CANN 9.0.0. This is not a
// hard compatibility gate: on another card or CANN version, revalidate the
// copied file and amend its compile-time assumptions locally when necessary.
// Extend only for a proved local need: one small declarative field/predicate
// plus compile-time positive and negative cases. Do not add runtime machinery.
// Every helper in this file is ordinary host-side constexpr C++. Use it in
// static_assert or host validation only. A device runtime formula must be
// inlined at its __aicore__ call site or implemented by a solution-local
// __aicore__ helper marked OPFORGE_DEVICE_INLINE_LAYOUT_FORMULA.
#define OPFORGE_D16_PAIR_INPUT_SUPPORTED(input_kind, fp16_kind, bf16_kind) \
    ((input_kind) == (fp16_kind) || (input_kind) == (bf16_kind))
#define OPFORGE_DUAL_PROBABILITY_WORKSPACE_VALID(extra_bytes, probability_bytes, accumulator_bytes) \
    ((extra_bytes) == (probability_bytes) + (accumulator_bytes) && (probability_bytes) > 0 && (accumulator_bytes) > 0)

static_assert(OPFORGE_D16_PAIR_INPUT_SUPPORTED(0, 0, 1), "FP16 must enable D16 pair compaction");
static_assert(OPFORGE_D16_PAIR_INPUT_SUPPORTED(1, 0, 1),
              "BF16 must enable D16 pair compaction after staged-V publication");
static_assert(!OPFORGE_D16_PAIR_INPUT_SUPPORTED(2, 0, 1),
              "unsupported input dtypes must not enable D16 pair compaction");
static_assert(OPFORGE_DUAL_PROBABILITY_WORKSPACE_VALID(6, 2, 4),
              "dual probability requires one FP16 residual and one FP32 PV accumulator per element");
static_assert(!OPFORGE_DUAL_PROBABILITY_WORKSPACE_VALID(4, 2, 4),
              "dual probability must reject an undersized residual workspace");

#ifndef OPFORGE_RESOURCE_LEDGER_HOST_PROJECTION_ONLY

#include <cstddef>
#include <cstdint>

namespace opforge::ascend_c::resource_ledger {

constexpr std::size_t kMaxSize = static_cast<std::size_t>(-1);
constexpr std::size_t kNoRegion = static_cast<std::size_t>(-1);

struct Region {
    std::uint64_t storage;
    std::size_t offset;
    std::size_t bytes;
    std::size_t alignment;
    std::uint32_t first;
    std::uint32_t last;
    std::size_t capacity = kMaxSize;
    std::uint32_t alias_group = 0;
};

constexpr bool AddWouldOverflow(std::size_t lhs, std::size_t rhs)
{
    return rhs > kMaxSize - lhs;
}

constexpr bool RangeFits(std::size_t offset, std::size_t bytes, std::size_t capacity)
{
    return offset <= capacity && !AddWouldOverflow(offset, bytes) && bytes <= capacity - offset;
}

constexpr bool RegionIsValid(const Region &region)
{
    return region.bytes != 0 && region.alignment != 0 && region.offset % region.alignment == 0 &&
           region.first < region.last && RangeFits(region.offset, region.bytes, region.capacity);
}

constexpr bool IntervalsOverlap(std::size_t a_begin, std::size_t a_bytes, std::size_t b_begin, std::size_t b_bytes)
{
    if (a_bytes == 0 || b_bytes == 0)
        return false;
    return a_begin < b_begin ? b_begin - a_begin < a_bytes : a_begin - b_begin < b_bytes;
}

constexpr bool ExactAlias(const Region &a, const Region &b)
{
    return a.alias_group != 0 && a.alias_group == b.alias_group && a.storage == b.storage && a.offset == b.offset &&
           a.bytes == b.bytes && a.capacity == b.capacity;
}

constexpr bool RegionsConflict(const Region &a, const Region &b)
{
    if (a.storage != b.storage || ExactAlias(a, b))
        return false;
    return a.first < b.last && b.first < a.last && IntervalsOverlap(a.offset, a.bytes, b.offset, b.bytes);
}

template <std::size_t N>
constexpr bool Validate(const Region (&regions)[N])
{
    for (std::size_t i = 0; i < N; ++i) {
        if (!RegionIsValid(regions[i]))
            return false;
        for (std::size_t j = i + 1; j < N; ++j) {
            if (regions[i].storage == regions[j].storage && regions[i].capacity != regions[j].capacity)
                return false;
            const bool same_alias = regions[i].alias_group != 0 && regions[i].alias_group == regions[j].alias_group;
            if (same_alias && !ExactAlias(regions[i], regions[j]))
                return false;
            if (RegionsConflict(regions[i], regions[j]))
                return false;
        }
    }
    return true;
}

struct Access {
    std::uint64_t storage;
    std::size_t offset;
    std::size_t logical_bytes;
    std::size_t effective_bytes;
    std::size_t capacity;
    std::size_t alignment;
    std::size_t region = kNoRegion;
};

constexpr bool ValidateAccess(const Access &access)
{
    return access.effective_bytes != 0 && access.alignment != 0 && access.offset % access.alignment == 0 &&
           access.effective_bytes >= access.logical_bytes &&
           RangeFits(access.offset, access.effective_bytes, access.capacity);
}

constexpr bool ValidateAccess(const Region &region, const Access &access)
{
    if (!RegionIsValid(region) || !ValidateAccess(access) || access.storage != region.storage ||
        access.capacity != region.capacity || access.offset < region.offset)
        return false;
    const std::size_t relative = access.offset - region.offset;
    return RangeFits(relative, access.effective_bytes, region.bytes);
}

template <std::size_t RegionCount, std::size_t AccessCount>
constexpr bool ValidateAccesses(const Region (&regions)[RegionCount], const Access (&accesses)[AccessCount])
{
    if (!Validate(regions))
        return false;
    for (std::size_t i = 0; i < AccessCount; ++i) {
        if (accesses[i].region >= RegionCount || !ValidateAccess(regions[accesses[i].region], accesses[i]))
            return false;
    }
    return true;
}

enum class L0Event : std::uint8_t {
    MTE1_M,
    M_MTE1,
};

constexpr bool ValidateL0ReuseEventPair(L0Event published, L0Event released)
{
    return published == L0Event::MTE1_M && released == L0Event::M_MTE1;
}

enum class L0cCompletionProtocol : std::uint8_t {
    UnitFlag,
    ExplicitMFixFixM,
};

struct L0cReuseContract {
    L0cCompletionProtocol producer;
    L0cCompletionProtocol next_writer;
    bool same_physical_owner;
    bool producer_releases;
    bool next_writer_waits;
};

constexpr bool ValidateL0cReuseContract(const L0cReuseContract &contract)
{
    return contract.same_physical_owner && contract.producer == contract.next_writer && contract.producer_releases &&
           contract.next_writer_waits;
}

enum class UbAccessRole : std::uint8_t {
    VectorReader,
    Mte2Writer,
};

enum class UbReuseEvent : std::uint8_t {
    V_MTE2,
    MTE2_V,
};

struct UbOverwriteContract {
    UbAccessRole prior;
    UbAccessRole next;
    UbReuseEvent publish;
    UbReuseEvent wait;
    bool same_physical_owner;
};

constexpr bool ValidateUbOverwriteContract(const UbOverwriteContract &contract)
{
    return contract.same_physical_owner && contract.prior == UbAccessRole::VectorReader &&
           contract.next == UbAccessRole::Mte2Writer && contract.publish == UbReuseEvent::V_MTE2 &&
           contract.wait == UbReuseEvent::V_MTE2;
}

constexpr UbOverwriteContract kVectorReadBeforeMte2Overwrite = {UbAccessRole::VectorReader, UbAccessRole::Mte2Writer,
                                                                UbReuseEvent::V_MTE2, UbReuseEvent::V_MTE2, true};
static_assert(ValidateUbOverwriteContract(kVectorReadBeforeMte2Overwrite));
static_assert(!ValidateUbOverwriteContract({UbAccessRole::VectorReader, UbAccessRole::Mte2Writer, UbReuseEvent::MTE2_V,
                                            UbReuseEvent::MTE2_V, true}),
              "a forward MTE2_V event cannot release a prior Vector reader");
static_assert(!ValidateUbOverwriteContract({UbAccessRole::VectorReader, UbAccessRole::Mte2Writer, UbReuseEvent::V_MTE2,
                                            UbReuseEvent::V_MTE2, false}),
              "a UB overwrite contract must name one reused physical owner");

// Bind this contract beside both the compile-time diagnostic guard and the
// exact resident-stage cut. Enabled cuts are diagnostic only and must close
// every local/cross-core payload and credit lifetime before participant exit.
struct DiagnosticStageCutContract {
    bool enabled;
    bool cut_reachable;
    bool producer_complete;
    bool no_live_payload;
    bool no_live_credit;
    bool all_participants_can_exit;
    bool incomplete_output_expected;
};

constexpr bool ValidateDiagnosticStageCut(const DiagnosticStageCutContract &contract)
{
    return !contract.enabled ||
           (contract.cut_reachable && contract.producer_complete && contract.no_live_payload &&
            contract.no_live_credit && contract.all_participants_can_exit && contract.incomplete_output_expected);
}

static_assert(ValidateDiagnosticStageCut({true, true, true, true, true, true, true}));
static_assert(ValidateDiagnosticStageCut({false, false, false, false, false, false, false}));
static_assert(!ValidateDiagnosticStageCut({true, false, true, true, true, true, true}),
              "an enabled stage cut must be reachable");
static_assert(!ValidateDiagnosticStageCut({true, true, true, false, true, true, true}),
              "a stage cut cannot leave a live payload");
static_assert(!ValidateDiagnosticStageCut({true, true, true, true, false, true, true}),
              "a stage cut cannot leave a live credit or acknowledgement");

template <typename ProducerLayout, typename ConsumerLayout>
struct SameLayoutTag {
    static constexpr bool value = false;
};

template <typename Layout>
struct SameLayoutTag<Layout, Layout> {
    static constexpr bool value = true;
};

template <typename ProducerLayout, typename ConsumerLayout>
constexpr bool ValidateLayoutHandoff()
{
    return SameLayoutTag<ProducerLayout, ConsumerLayout>::value;
}

enum class LayoutOwnerRole : std::uint8_t {
    Producer,
    Consumer
};
enum class AttentionPayloadRole : std::uint8_t {
    P,
    V
};

struct TypedPartLayout {
    LayoutOwnerRole owner;
    AttentionPayloadRole payload;
    std::size_t rows;
    std::size_t cols;
    std::size_t element_bytes;
    std::size_t part_count;
    std::size_t part_stride_bytes;
    std::size_t owner_bytes;
};

constexpr bool MultiplyWouldOverflow(std::size_t lhs, std::size_t rhs)
{
    return lhs != 0U && rhs > kMaxSize / lhs;
}

constexpr std::size_t DensePartBytes(std::size_t rows, std::size_t cols, std::size_t element_bytes)
{
    if (rows == 0U || cols == 0U || element_bytes == 0U || MultiplyWouldOverflow(rows, cols))
        return kMaxSize;
    const std::size_t elements = rows * cols;
    return MultiplyWouldOverflow(elements, element_bytes) ? kMaxSize : elements * element_bytes;
}

constexpr bool ValidateTypedPartLayout(const TypedPartLayout &layout)
{
    const std::size_t part_bytes = DensePartBytes(layout.rows, layout.cols, layout.element_bytes);
    return part_bytes != kMaxSize && layout.part_count > 0U && layout.part_stride_bytes == part_bytes &&
           !MultiplyWouldOverflow(layout.part_count, layout.part_stride_bytes) &&
           layout.owner_bytes == layout.part_count * layout.part_stride_bytes;
}

constexpr bool ValidateTypedLayoutHandoff(const TypedPartLayout &producer, const TypedPartLayout &consumer)
{
    return producer.owner == LayoutOwnerRole::Producer && consumer.owner == LayoutOwnerRole::Consumer &&
           producer.payload == consumer.payload && producer.rows == consumer.rows && producer.cols == consumer.cols &&
           producer.element_bytes == consumer.element_bytes && producer.part_count == consumer.part_count &&
           producer.part_stride_bytes == consumer.part_stride_bytes && producer.owner_bytes == consumer.owner_bytes &&
           ValidateTypedPartLayout(producer) && ValidateTypedPartLayout(consumer);
}

constexpr std::size_t TypedPartOffset(const TypedPartLayout &layout, std::size_t part)
{
    return ValidateTypedPartLayout(layout) && part < layout.part_count ? part * layout.part_stride_bytes : kMaxSize;
}

constexpr TypedPartLayout kPProducerLayout = {
    LayoutOwnerRole::Producer, AttentionPayloadRole::P, 128U, 64U, 1U, 4U, 8192U, 32768U};
constexpr TypedPartLayout kPConsumerLayout = {
    LayoutOwnerRole::Consumer, AttentionPayloadRole::P, 128U, 64U, 1U, 4U, 8192U, 32768U};
constexpr TypedPartLayout kVProducerD64Layout = {
    LayoutOwnerRole::Producer, AttentionPayloadRole::V, 64U, 64U, 1U, 4U, 4096U, 16384U};
constexpr TypedPartLayout kVConsumerD64Layout = {
    LayoutOwnerRole::Consumer, AttentionPayloadRole::V, 64U, 64U, 1U, 4U, 4096U, 16384U};
constexpr TypedPartLayout kVProducerD128Layout = {
    LayoutOwnerRole::Producer, AttentionPayloadRole::V, 64U, 128U, 1U, 4U, 8192U, 32768U};
constexpr TypedPartLayout kVConsumerD128Layout = {
    LayoutOwnerRole::Consumer, AttentionPayloadRole::V, 64U, 128U, 1U, 4U, 8192U, 32768U};
static_assert(ValidateTypedLayoutHandoff(kPProducerLayout, kPConsumerLayout));
static_assert(ValidateTypedLayoutHandoff(kVProducerD64Layout, kVConsumerD64Layout));
static_assert(ValidateTypedLayoutHandoff(kVProducerD128Layout, kVConsumerD128Layout));
static_assert(TypedPartOffset(kVProducerD64Layout, 3U) == 12288U);
static_assert(TypedPartOffset(kVProducerD128Layout, 3U) == 24576U);
static_assert(!ValidateTypedPartLayout({LayoutOwnerRole::Consumer, AttentionPayloadRole::V, 64U, 64U, 1U, 4U, 4097U,
                                        16388U}),
              "a D64 V consumer must reject a one-byte-past part stride");
static_assert(!ValidateTypedLayoutHandoff(kPProducerLayout, kVConsumerD128Layout),
              "P and V payload roles cannot share a stride contract");

namespace protocol {

struct DualProbabilityNumericContract {
    bool fp32_probability_owned;
    bool fp16_main_written;
    bool scaled_residual_written;
    bool main_pv_owned;
    bool residual_pv_owned;
    bool fp32_addback_owned;
    bool host_workspace_owned;
    float residual_scale;
    float addback_scale;
};

constexpr bool ValidateDualProbabilityNumericContract(const DualProbabilityNumericContract &contract)
{
    return contract.fp32_probability_owned && contract.fp16_main_written && contract.scaled_residual_written &&
           contract.main_pv_owned && contract.residual_pv_owned && contract.fp32_addback_owned &&
           contract.host_workspace_owned && contract.residual_scale > 0.0f && contract.addback_scale > 0.0f;
}

static_assert(ValidateDualProbabilityNumericContract({true, true, true, true, true, true, true, 4096.0f,
                                                      0.5f / 4096.0f}));
static_assert(!ValidateDualProbabilityNumericContract({true, true, true, true, true, true, false, 4096.0f,
                                                       0.5f / 4096.0f}),
              "dual probability must reject a producer/consumer chain without host-owned residual workspace");
static_assert(!ValidateDualProbabilityNumericContract({true, true, false, true, true, true, true, 4096.0f,
                                                       0.5f / 4096.0f}),
              "dual probability must reject a chain that never publishes the residual payload");

enum class D16PairInputKind : std::uint8_t {
    Fp16,
    Bf16,
};

constexpr bool ValidateD16PairInputKind(D16PairInputKind input_kind)
{
    return OPFORGE_D16_PAIR_INPUT_SUPPORTED(input_kind, D16PairInputKind::Fp16, D16PairInputKind::Bf16);
}

constexpr bool ValidateD16PairStagingReachability(D16PairInputKind input_kind, bool staged_v_collective_reachable)
{
    return ValidateD16PairInputKind(input_kind) &&
           (input_kind != D16PairInputKind::Bf16 || staged_v_collective_reachable);
}

static_assert(ValidateD16PairInputKind(D16PairInputKind::Fp16));
static_assert(ValidateD16PairStagingReachability(D16PairInputKind::Fp16, false));
static_assert(ValidateD16PairStagingReachability(D16PairInputKind::Bf16, true));
static_assert(!ValidateD16PairStagingReachability(D16PairInputKind::Bf16, false),
              "BF16 pair requires staged-V collective reachability");

constexpr std::size_t OutstandingTokenWindow(std::size_t query_count, std::size_t prelaunch_depth)
{
    return query_count == 0U || prelaunch_depth == kMaxSize ? kMaxSize : query_count * (prelaunch_depth + 1U);
}

constexpr bool ValidateCrossCorePrelaunch(std::size_t query_count, std::size_t prelaunch_depth,
                                          std::size_t owned_token_budget)
{
    const std::size_t outstanding = OutstandingTokenWindow(query_count, prelaunch_depth);
    return query_count > 0U && owned_token_budget > 0U && outstanding != kMaxSize && outstanding <= owned_token_budget;
}

constexpr std::size_t kD16PairQueryCount = 6U;
constexpr std::size_t kD16PairPrelaunchDepth = 1U;
constexpr std::size_t kD16PairOwnedTokenBudget = 12U;
static_assert(ValidateCrossCorePrelaunch(kD16PairQueryCount, kD16PairPrelaunchDepth, kD16PairOwnedTokenBudget));
static_assert(!ValidateCrossCorePrelaunch(kD16PairQueryCount, 2U, kD16PairOwnedTokenBudget),
              "D16 pair depth two exceeds its owned token window");

constexpr std::size_t kD16GroupedTextQueryRows = 128U;
constexpr std::size_t kD16GroupedTextMaxQueryCount = 4U;
constexpr std::size_t kD16GroupedKernelQueryCapacity = 6U;

constexpr bool ValidateAivPhysicalWorkspaceOwner(std::size_t logical_block_idx, std::size_t subblock_count,
                                                 std::size_t physical_core_count)
{
    return subblock_count > 0U && physical_core_count > 0U && logical_block_idx / subblock_count < physical_core_count;
}

static_assert(ValidateAivPhysicalWorkspaceOwner(47U, 2U, 24U));
static_assert(!ValidateAivPhysicalWorkspaceOwner(48U, 2U, 24U),
              "logical AIV blocks must map inside the physical workspace owner range");
static_assert(!ValidateAivPhysicalWorkspaceOwner(0U, 0U, 24U),
              "AIV workspace owner normalization requires a nonzero subblock count");

constexpr bool ValidateUbResidentState(std::size_t query_count, std::size_t ring_stages, std::size_t row_elements,
                                       std::size_t element_bytes, std::size_t capacity_bytes)
{
    if (query_count == 0U || ring_stages == 0U || row_elements == 0U || element_bytes == 0U) {
        return false;
    }
    const std::size_t fields = 2U + ring_stages;
    if (query_count > kMaxSize / fields || query_count * fields > kMaxSize / row_elements ||
        query_count * fields * row_elements > kMaxSize / element_bytes) {
        return false;
    }
    return query_count * fields * row_elements * element_bytes <= capacity_bytes;
}

constexpr bool ValidateUbResidentStateWithPayload(std::size_t query_count, std::size_t ring_stages,
                                                  std::size_t row_elements, std::size_t payload_elements_per_query,
                                                  std::size_t element_bytes, std::size_t capacity_bytes)
{
    if (!ValidateUbResidentState(query_count, ring_stages, row_elements, element_bytes, capacity_bytes) ||
        payload_elements_per_query == 0U || query_count > kMaxSize / payload_elements_per_query) {
        return false;
    }
    const std::size_t state_elements = query_count * (2U + ring_stages) * row_elements;
    const std::size_t payload_elements = query_count * payload_elements_per_query;
    if (state_elements > kMaxSize - payload_elements) {
        return false;
    }
    const std::size_t total_elements = state_elements + payload_elements;
    return total_elements <= kMaxSize / element_bytes && total_elements * element_bytes <= capacity_bytes;
}

constexpr bool ValidateUbResidentPayload(std::size_t query_count, std::size_t elements_per_query,
                                         std::size_t element_bytes, std::size_t capacity_bytes)
{
    return query_count > 0U && elements_per_query > 0U && element_bytes > 0U &&
           query_count <= kMaxSize / elements_per_query &&
           query_count * elements_per_query <= kMaxSize / element_bytes &&
           query_count * elements_per_query * element_bytes <= capacity_bytes;
}

static_assert(ValidateUbResidentState(6U, 5U, 64U, sizeof(float), 16U * 1024U));
static_assert(!ValidateUbResidentState(10U, 5U, 64U, sizeof(float), 16U * 1024U),
              "ten query chunks exceed the pair UB state region");
static_assert(ValidateUbResidentState(3U, 5U, 64U, sizeof(float), 16U * 1024U));
static_assert(ValidateUbResidentPayload(2U, 64U * 32U, sizeof(float), 16U * 1024U));
static_assert(ValidateUbResidentPayload(1U, 64U * 32U, sizeof(float), 16U * 1024U));
static_assert(!ValidateUbResidentPayload(2U, 64U * 64U, sizeof(float), 16U * 1024U),
              "two D64 Go payloads exceed one grouped UB payload owner");

constexpr bool ValidateD16GroupedTextOwnership(std::size_t text_length, std::size_t text_storage_capacity,
                                               std::size_t query_rows = kD16GroupedTextQueryRows)
{
    const std::size_t query_count = query_rows == 0U ? kMaxSize : text_storage_capacity / query_rows;
    return query_rows > 0U && text_length > query_rows && text_length <= text_storage_capacity &&
           text_storage_capacity % query_rows == 0U && query_count >= 2U &&
           query_count <= kD16GroupedTextMaxQueryCount && query_count <= kD16GroupedKernelQueryCapacity &&
           ValidateCrossCorePrelaunch(query_count, kD16PairPrelaunchDepth, kD16PairOwnedTokenBudget);
}

static_assert(ValidateD16GroupedTextOwnership(239U, 256U));
static_assert(ValidateD16GroupedTextOwnership(512U, 512U));
static_assert(!ValidateD16GroupedTextOwnership(128U, 128U), "one text query chunk must stay on the short-text path");
static_assert(!ValidateD16GroupedTextOwnership(161U, 161U),
              "a grouped text tail requires storage for its rounded Q load");
static_assert(!ValidateD16GroupedTextOwnership(513U, 640U),
              "the task contract and grouped state own at most four text queries");

} // namespace protocol

static_assert(ValidateL0ReuseEventPair(L0Event::MTE1_M, L0Event::M_MTE1));
static_assert(!ValidateL0ReuseEventPair(L0Event::M_MTE1, L0Event::MTE1_M));
static_assert(ValidateL0cReuseContract({L0cCompletionProtocol::ExplicitMFixFixM,
                                        L0cCompletionProtocol::ExplicitMFixFixM, true, true, true}));
static_assert(!ValidateL0cReuseContract({L0cCompletionProtocol::ExplicitMFixFixM, L0cCompletionProtocol::UnitFlag, true,
                                         true, true}));

template <const auto &Regions>
struct StaticLedger {
    static constexpr std::size_t size = sizeof(Regions) / sizeof(Regions[0]);
    static constexpr bool valid = Validate(Regions);

    template <std::size_t Id>
    static constexpr Region Get()
    {
        static_assert(Id < size, "StaticLedger region is outside the ledger");
        static_assert(valid, "StaticLedger requires a valid complete ledger");
        return Regions[Id];
    }
};

template <std::size_t Id, typename Ledger>
constexpr std::size_t CheckedOffset(Ledger)
{
    return Ledger::template Get<Id>().offset;
}

template <std::size_t Id, typename Ledger>
constexpr std::size_t CheckedBytes(Ledger)
{
    return Ledger::template Get<Id>().bytes;
}

} // namespace opforge::ascend_c::resource_ledger

// GM ownership is part of the same Ascend-C resource ledger.  This host/device
// companion validates allocation intervals and monotonic tiled footprints;
// keeping it here prevents a second, parallel checker contract.
namespace opforge::ascend_c::resource_ledger::gm {
enum class Access : std::uint8_t {
    Read,
    Write
};
enum class Error : std::uint8_t {
    Ok,
    ZeroExtent,
    ZeroAlignment,
    MisalignedAddress,
    ArithmeticOverflow,
    AccessOutsideOwner,
    InvalidTile,
    InvalidLaunch
};
struct Result {
    Error error;
    std::uint32_t logical_task;
    std::uint32_t k_iteration;
    Access access;
    constexpr bool ok() const
    {
        return error == Error::Ok;
    }
};
struct RuntimeOwner {
    std::uintptr_t base;
    std::size_t bytes;
    std::size_t alignment;
};
constexpr std::size_t kInvalidSize = static_cast<std::size_t>(-1);
constexpr bool AddWouldOverflow(std::size_t a, std::size_t b)
{
    return b > kInvalidSize - a;
}
constexpr bool MultiplyWouldOverflow(std::size_t a, std::size_t b)
{
    return a != 0U && b > kInvalidSize / a;
}
constexpr std::size_t Mul(std::size_t a, std::size_t b)
{
    return MultiplyWouldOverflow(a, b) ? kInvalidSize : a * b;
}
constexpr std::size_t Add(std::size_t a, std::size_t b)
{
    return AddWouldOverflow(a, b) ? kInvalidSize : a + b;
}
constexpr std::size_t Span(std::size_t rows, std::size_t cols, std::size_t stride, std::size_t elem)
{
    if (!rows || !cols || !elem || stride < cols)
        return kInvalidSize;
    return Mul(Add(Mul(rows - 1U, stride), cols), elem);
}
constexpr bool Fits(std::size_t off, std::size_t span, std::size_t owner)
{
    return span != 0U && span != kInvalidSize && off <= owner && !AddWouldOverflow(off, span) && span <= owner - off;
}
constexpr Result ValidateAccess(const RuntimeOwner &owner, std::uintptr_t address, std::size_t span, Access access,
                                std::uint32_t task, std::uint32_t ki)
{
    if (!owner.bytes || !span)
        return {Error::ZeroExtent, task, ki, access};
    if (!owner.alignment)
        return {Error::ZeroAlignment, task, ki, access};
    if (owner.base % owner.alignment || address % owner.alignment)
        return {Error::MisalignedAddress, task, ki, access};
    if (address < owner.base || span == kInvalidSize)
        return {span == kInvalidSize ? Error::ArithmeticOverflow : Error::AccessOutsideOwner, task, ki, access};
    const std::uintptr_t delta = address - owner.base;
    return delta > kInvalidSize || !Fits(static_cast<std::size_t>(delta), span, owner.bytes) ?
               Result{Error::AccessOutsideOwner, task, ki, access} :
               Result{Error::Ok, task, ki, access};
}
template <std::size_t E, std::size_t K, std::size_t N, std::size_t R, std::size_t TR, std::size_t TK, std::size_t TN>
struct StaticLinearContract {
    static constexpr std::size_t in_bytes = Mul(Mul(R, K), E), out_bytes = Mul(Mul(R, N), E),
                                 weight_bytes = Mul(Mul(N, K), E);
    static constexpr std::size_t input_owner_bytes = in_bytes;
    static constexpr std::size_t output_owner_bytes = out_bytes;
    static constexpr std::size_t weight_owner_bytes = weight_bytes;
    static constexpr bool valid = E && K && N && R >= TR && TR && TK && TN && K % TK == 0U && N % TN == 0U &&
                                  Fits(Mul(K - TK, E), Span(TR, TK, K, E), in_bytes) &&
                                  Fits(Mul(Add(Mul(N - TN, K), K - TK), E), Span(TN, TK, K, E), weight_bytes) &&
                                  Fits(Mul(N - TN, E), Span(TR, TN, N, E), out_bytes);
};
template <typename C>
struct CheckedStaticLinearContract {
    static_assert(C::valid, "GM contract rejects linear tile");
    using type = C;
};
inline Result ValidateLinearLaunch(const RuntimeOwner &out, std::uintptr_t outp, const RuntimeOwner &in,
                                   std::uintptr_t inp, const RuntimeOwner &w, std::uintptr_t wp, std::uint32_t rows,
                                   std::uint32_t physical_rows, std::uint32_t K, std::uint32_t N, std::uint32_t TR,
                                   std::uint32_t TK, std::uint32_t TN, std::uint32_t E, std::uint32_t blocks)
{
    if (!rows || physical_rows < rows || !TR || !TK || !TN || !E || !K || !N || K % TK || N % TN)
        return {Error::InvalidTile, 0, 0, Access::Read};
    const std::uint32_t rt = (rows + TR - 1U) / TR, nt = N / TN;
    if (!blocks || blocks > rt * nt)
        return {Error::InvalidLaunch, 0, 0, Access::Read};
    if (in.bytes < Mul(Mul(physical_rows, K), E) || out.bytes < Mul(Mul(physical_rows, N), E))
        return {Error::AccessOutsideOwner, 0, 0, Access::Read};
    const std::uint32_t task = rt * nt, ki = K / TK;
    const std::size_t ro = static_cast<std::size_t>(rt - 1U) * TR, no = static_cast<std::size_t>(nt - 1U) * TN,
                      ko = static_cast<std::size_t>(ki - 1U) * TK;
    Result r =
        ValidateAccess(in, inp + Mul(Add(Mul(ro, K), ko), E), Span(TR, TK, K, E), Access::Read, task - 1U, ki - 1U);
    if (!r.ok())
        return r;
    r = ValidateAccess(w, wp + Mul(Add(Mul(no, K), ko), E), Span(TN, TK, K, E), Access::Read, task - 1U, ki - 1U);
    if (!r.ok())
        return r;
    return ValidateAccess(out, outp + Mul(Add(Mul(ro, N), no), E), Span(TR, TN, N, E), Access::Write, task - 1U, ki);
}
} // namespace opforge::ascend_c::resource_ledger::gm

#endif // OPFORGE_RESOURCE_LEDGER_HOST_PROJECTION_ONLY
