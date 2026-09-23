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

constexpr bool AddWouldOverflow(std::size_t lhs, std::size_t rhs) { return rhs > kMaxSize - lhs; }

constexpr bool RangeFits(std::size_t offset, std::size_t bytes, std::size_t capacity)
{
    return offset <= capacity && !AddWouldOverflow(offset, bytes) && bytes <= capacity - offset;
}

constexpr bool RegionIsValid(const Region& region)
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

constexpr bool ExactAlias(const Region& a, const Region& b)
{
    return a.alias_group != 0 && a.alias_group == b.alias_group && a.storage == b.storage && a.offset == b.offset &&
           a.bytes == b.bytes && a.capacity == b.capacity;
}

constexpr bool RegionsConflict(const Region& a, const Region& b)
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

constexpr bool ValidateAccess(const Access& access)
{
    return access.effective_bytes != 0 && access.alignment != 0 && access.offset % access.alignment == 0 &&
           access.effective_bytes >= access.logical_bytes &&
           RangeFits(access.offset, access.effective_bytes, access.capacity);
}

constexpr bool ValidateAccess(const Region& region, const Access& access)
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

constexpr bool ValidateL0cReuseContract(const L0cReuseContract& contract)
{
    return contract.same_physical_owner && contract.producer == contract.next_writer && contract.producer_releases &&
           contract.next_writer_waits;
}

enum class UbAccessRole : std::uint8_t {
    VectorReader,
    VectorWriter,
    Mte3Reader,
    Mte2Writer,
};

enum class UbReuseEvent : std::uint8_t {
    V_MTE2,
    MTE2_V,
    MTE3_V,
    MTE3_MTE2,
};

struct UbOverwriteContract {
    UbAccessRole prior;
    UbAccessRole next;
    UbReuseEvent publish;
    UbReuseEvent wait;
    bool same_physical_owner;
};

// One row per legal UB reuse edge. A reuse edge is only closed when the
// *prior* accessor's own pipe is drained before the next writer starts: a
// `V_MTE2` event releases a Vector reader, and an `MTE3_MTE2` event releases an
// MTE3 reader. Using the wrong event leaves the prior access in flight, so the
// row is data and an unknown edge fails closed.
struct UbReuseEdgeFact {
    UbAccessRole prior;
    UbAccessRole next;
    UbReuseEvent publish;
    UbReuseEvent wait;
};

constexpr UbReuseEdgeFact kUbReuseEdgeFacts[] = {
    {UbAccessRole::VectorReader, UbAccessRole::Mte2Writer, UbReuseEvent::V_MTE2, UbReuseEvent::V_MTE2},
    {UbAccessRole::Mte3Reader, UbAccessRole::VectorWriter, UbReuseEvent::MTE3_V, UbReuseEvent::MTE3_V},
};

constexpr bool ValidateUbOverwriteContract(const UbOverwriteContract& contract)
{
    if (!contract.same_physical_owner)
        return false;
    for (std::size_t i = 0; i < sizeof(kUbReuseEdgeFacts) / sizeof(kUbReuseEdgeFacts[0]); ++i) {
        if (kUbReuseEdgeFacts[i].prior == contract.prior && kUbReuseEdgeFacts[i].next == contract.next)
            return kUbReuseEdgeFacts[i].publish == contract.publish && kUbReuseEdgeFacts[i].wait == contract.wait;
    }
    return false;
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

// The rescale epilogue's output write (`CopyOToGm`, MTE3) reads `goUbTensor16`,
// and the next rescale's input load (`DataCopy(goUbTensor32, gInput)`, MTE2)
// writes the same UB address. The output write happens only on a task's last
// stack, so this is a task-boundary reuse edge. The Vector reads of the same
// owner (`Div`/`Cast`) sit *after* the epilogue's `V_MTE2` publication, so the
// edge is closed by draining the MTE3 into the Vector pipe first and only then
// publishing `V_MTE2`; the next rescale's `V_MTE2` wait then transitively
// orders its MTE2 overwrite after both readers.
constexpr UbOverwriteContract kRescaleOutputReadBeforeVectorPublication = {
    UbAccessRole::Mte3Reader, UbAccessRole::VectorWriter, UbReuseEvent::MTE3_V, UbReuseEvent::MTE3_V, true};
static_assert(ValidateUbOverwriteContract(kRescaleOutputReadBeforeVectorPublication));
static_assert(!ValidateUbOverwriteContract({UbAccessRole::Mte3Reader, UbAccessRole::VectorWriter, UbReuseEvent::V_MTE2,
                                            UbReuseEvent::V_MTE2, true}),
              "a V_MTE2 event cannot release a prior MTE3 reader");
static_assert(!ValidateUbOverwriteContract({UbAccessRole::Mte3Reader, UbAccessRole::VectorWriter,
                                            UbReuseEvent::MTE3_V, UbReuseEvent::MTE3_V, false}),
              "the rescale output reuse edge must name one reused physical owner");

// The D16 narrow-score stage is rewritten by MTE2 at the head of every row-loop
// iteration and read by the vector upcast in between, so its reuse edge is a
// Vector-reader to next-MTE2-writer release carried on V_MTE2. Releasing it on
// MTE2_V instead would let the next copy start before the upcast read the tile.
constexpr UbOverwriteContract kNarrowScoreStageReuse = {UbAccessRole::VectorReader, UbAccessRole::Mte2Writer,
                                                        UbReuseEvent::V_MTE2, UbReuseEvent::V_MTE2, true};
static_assert(ValidateUbOverwriteContract(kNarrowScoreStageReuse));
static_assert(!ValidateUbOverwriteContract({UbAccessRole::VectorReader, UbAccessRole::Mte2Writer,
                                            UbReuseEvent::V_MTE2, UbReuseEvent::MTE2_V, true}),
              "the staged narrow score must be released to its next MTE2 writer on V_MTE2");

// ---------------------------------------------------------------------------
// AIV-stream event-slot ownership.
//
// Hardware event ids are a single per-core flag space shared by every event
// type. The online-softmax row loop runs a steady-state ping-pong on slots 0
// and 1 (`pingpongFlag = rowLoopIdx % 2`) and leaves one V_MTE2 flag armed on
// each of them for the next call. A `SetFlag`/`WaitFlag` pair on the same slot
// from another function is therefore satisfied by that leftover instead of by
// its own producer, which silently drops the intended ordering. The rescale
// epilogue must own slots that the online-softmax ping-pong never touches.
constexpr std::uint8_t kOnlineSoftmaxPingpongSlot0 = 0;
constexpr std::uint8_t kOnlineSoftmaxPingpongSlot1 = 1;
constexpr std::uint8_t kOnlineSoftmaxMaskSlot = 2;
constexpr std::uint8_t kRescaleCrossCallVMte2Slot = 3;
constexpr std::uint8_t kRescaleIntraCallSlot = 4;

constexpr bool RescaleEventSlotsAreIsolated(std::uint8_t intraCallSlot, std::uint8_t crossCallSlot)
{
    return intraCallSlot != kOnlineSoftmaxPingpongSlot0 && intraCallSlot != kOnlineSoftmaxPingpongSlot1 &&
           intraCallSlot != kOnlineSoftmaxMaskSlot && crossCallSlot != kOnlineSoftmaxPingpongSlot0 &&
           crossCallSlot != kOnlineSoftmaxPingpongSlot1 && crossCallSlot != kOnlineSoftmaxMaskSlot &&
           intraCallSlot != crossCallSlot;
}

static_assert(RescaleEventSlotsAreIsolated(kRescaleIntraCallSlot, kRescaleCrossCallVMte2Slot));
static_assert(!RescaleEventSlotsAreIsolated(kOnlineSoftmaxPingpongSlot0, kRescaleCrossCallVMte2Slot),
              "the rescale intra-call slot must not alias an online-softmax ping-pong slot");
static_assert(!RescaleEventSlotsAreIsolated(kRescaleIntraCallSlot, kOnlineSoftmaxMaskSlot),
              "the rescale cross-call slot must not alias the online-softmax mask slot");

// The D16 narrow-score bridge stages one fp16 score row tile and upcasts it inside
// the same row-loop iteration, so the staged owner must be released to the next
// iteration's copy by a dedicated V->MTE2 edge. That edge needs its own slot: a
// shared slot would be satisfied by another owner's leftover flag and silently
// drop the ordering.
constexpr std::uint8_t kNarrowScoreStageVMte2Slot = 5;

constexpr bool NarrowScoreStageSlotIsIsolated(std::uint8_t stageSlot)
{
    return stageSlot < 8U && stageSlot != kOnlineSoftmaxPingpongSlot0 && stageSlot != kOnlineSoftmaxPingpongSlot1 &&
           stageSlot != kOnlineSoftmaxMaskSlot && stageSlot != kRescaleCrossCallVMte2Slot &&
           stageSlot != kRescaleIntraCallSlot;
}

static_assert(NarrowScoreStageSlotIsIsolated(kNarrowScoreStageVMte2Slot));
static_assert(!NarrowScoreStageSlotIsIsolated(kOnlineSoftmaxPingpongSlot0),
              "the narrow-score stage slot must not alias an online-softmax ping-pong slot");
static_assert(!NarrowScoreStageSlotIsIsolated(kRescaleIntraCallSlot),
              "the narrow-score stage slot must not alias the rescale intra-call slot");
static_assert(!NarrowScoreStageSlotIsIsolated(8U), "the narrow-score stage slot must be a real event id");

// ---------------------------------------------------------------------------
// AIV-stream softmax lookahead.
//
// `ProcessAivSingleQueryPackedD32Stream` walks the same per-core stack sequence
// as `ProcessAicSingleQueryPackedD32Stream`. Round `r` writes the ring entry for
// stack `r`, then reads the entry for stack `r - kSoftmaxLookaheadDepth` (the
// online softmax) and the entry for stack `r - kRescaleLookaheadDepth` (the
// rescale). The three indices must land on three distinct ring slots, and each
// read must still be separated from the next write to its own slot, or the
// `StackInfo` metadata and the score/P/oTmp slot payload are reused while their
// consumer is still running.
//
// With `kSoftmaxLookaheadDepth == 0` the online softmax of stack `r` can only
// start after the AIC has published the whole `QK^T(r)`, which puts a serial
// `QK^T` term in front of the AIV's vector work every round. One round of
// lookahead removes that term: `softmax(r - 1)` then overlaps `QK^T(r)`.
//
// The `qkReady` and `softmaxReady` FFTS flags are counters with one credit per
// stack on each side, so a lookahead of `kSoftmaxLookaheadDepth` rounds simply
// leaves that many credits outstanding; the slot-disjointness predicate below is
// what keeps the payload safe.
constexpr std::uint32_t kSoftmaxLookaheadDepth = 1;
constexpr std::uint32_t kRescaleLookaheadDepth = 4;
constexpr std::uint32_t kStreamRingStageCount = 5;
constexpr std::uint32_t kQkReadyCreditsPerStack = 1;
constexpr std::uint32_t kSoftmaxReadyCreditsPerStack = 1;

constexpr bool StreamLookaheadSlotsAreDisjoint(std::uint32_t ringStageCount, std::uint32_t softmaxLookahead,
                                               std::uint32_t rescaleLookahead)
{
    return softmaxLookahead >= 1 && softmaxLookahead < ringStageCount && rescaleLookahead > softmaxLookahead &&
           rescaleLookahead < ringStageCount && (rescaleLookahead - softmaxLookahead) < ringStageCount;
}

static_assert(kQkReadyCreditsPerStack == kSoftmaxReadyCreditsPerStack,
              "the qkReady and softmaxReady edges must carry the same per-stack credit count");
static_assert(StreamLookaheadSlotsAreDisjoint(kStreamRingStageCount, kSoftmaxLookaheadDepth, kRescaleLookaheadDepth));
static_assert(!StreamLookaheadSlotsAreDisjoint(kStreamRingStageCount, 0, kRescaleLookaheadDepth),
              "a zero softmax lookahead reads the slot the same round writes");
static_assert(!StreamLookaheadSlotsAreDisjoint(kStreamRingStageCount, kSoftmaxLookaheadDepth, kStreamRingStageCount),
              "a rescale lookahead equal to the ring depth reads the slot the same round writes");
static_assert(!StreamLookaheadSlotsAreDisjoint(kStreamRingStageCount, kRescaleLookaheadDepth, kRescaleLookaheadDepth),
              "the softmax and rescale lookaheads must not share one ring slot");

// ---------------------------------------------------------------------------
// UB->UB handoff copy primitive: the pipe model is annotation-driven.
//
// A `PipeBarrier<PIPE_X>` only orders a call against the X pipe when the
// *caller-side* declaration of that call carries an `__inout_pipe__(X)`
// annotation. The annotation is invisible at the call site, so an unannotated
// primitive silently escapes the barrier and the compiler may reorder it
// across it. This is the implicit rule that cost this solution a 6.5 ms
// `PIPE_ALL` on case42.
//
// The rule is therefore recorded as data, not as prose: one row per primitive
// this solution may use, with the exact installed-CANN source reference and
// the intrinsic it lowers to. `UbCopyPrimitiveIsPipeModeled` is the only
// reader, so a new primitive must add a row, and the row must be re-verified
// against the active CANN root. `probe/verify_cann_pipe_annotations.py`
// re-checks every row against `OPFORGE_CANN_ROOT` and fails closed on drift.
//
// Verified against OPFORGE_CANN_ROOT=<cann-9.0.0> (ascend910_93):
//   * `DataCopy(LocalTensor, LocalTensor, DataCopyParams)` at
//     `asc/impl/basic_api/kernel_operator_data_copy_intf_impl.h:320` has NO
//     caller-side `__inout_pipe__`; the count form at `:756` and the Nd2Nz
//     form at `:224` are unannotated too. The impl `DataCopyUB2UBIntf` carries
//     only a region-level `#pragma begin_pipe(V)`
//     (`dav_c220/kernel_operator_data_copy_impl.h:1031-1039`), which is not
//     the caller-side annotation the barrier model reads.
//   * `Muls`/`Adds` are ALSO unannotated at the caller side
//     (`asc/include/basic_api/kernel_operator_vec_binary_scalar_intf.h:53-175`
//     contains zero `__inout_pipe__`); they carry only a file-level
//     `#pragma begin_pipe(V)` in the impl (`..._intf_impl.h:40`). They are
//     therefore NOT safe handoff primitives under `PipeBarrier<PIPE_V>`.
//   * `Copy` is the only UB->UB copy primitive with a caller-side
//     `__inout_pipe__(V)`
//     (`asc/include/basic_api/kernel_operator_data_copy_intf.h:172,178,185`)
//     and lowers to `vcopy` (`dav_c220/..._impl.h:339-375`).
enum class UbHandoffCopyPrimitive : std::uint8_t {
    DataCopyUbToUb,
    VCopy,
    VMuls,
    VAdds,
};

// Caller-side `__inout_pipe__` annotation actually present on the installed
// CANN declaration. `None` means the barrier model cannot see the call.
enum class UbPipeAnnotation : std::uint8_t {
    None,
    V,
    MTE2,
    MTE3,
};

// The mask-mode `Copy` overloads differ in what the `mask` parameter means.
// The header documents it as `@param [in] mask[]/mask mask array/count`, and
// `AscendCUtils::SetMask<T>(int32_t len)`
// (`asc/impl/basic_api/kernel_utils_base.h:126`) builds the hardware mask from
// that element count. Passing a bitmask to the count form silently produces a
// garbage mask (`len == -1`), so the semantics are part of the fact row.
enum class UbCopyMaskSemantics : std::uint8_t {
    NotApplicable,  // block-form copy: no mask parameter
    ElementCount,   // `uint64_t mask` = number of elements per repeat
    BitMask,        // `uint64_t mask[]` = per-element bit mask
};

struct UbCopyPrimitiveFact {
    UbHandoffCopyPrimitive primitive;
    UbPipeAnnotation caller_annotation;
    UbCopyMaskSemantics mask_semantics;
    const char* lowers_to;
    const char* source_ref;
};

constexpr UbCopyPrimitiveFact kUbCopyPrimitiveFacts[] = {
    {UbHandoffCopyPrimitive::DataCopyUbToUb, UbPipeAnnotation::None, UbCopyMaskSemantics::NotApplicable,
     "copy_ubuf_to_ubuf",
     "asc/impl/basic_api/kernel_operator_data_copy_intf_impl.h:320"},
    {UbHandoffCopyPrimitive::VCopy, UbPipeAnnotation::V, UbCopyMaskSemantics::ElementCount, "vcopy",
     "asc/include/basic_api/kernel_operator_data_copy_intf.h:172"},
    {UbHandoffCopyPrimitive::VMuls, UbPipeAnnotation::None, UbCopyMaskSemantics::NotApplicable, "vmuls",
     "asc/include/basic_api/kernel_operator_vec_binary_scalar_intf.h:124"},
    {UbHandoffCopyPrimitive::VAdds, UbPipeAnnotation::None, UbCopyMaskSemantics::NotApplicable, "vadds",
     "asc/include/basic_api/kernel_operator_vec_binary_scalar_intf.h:53"},
};

// Unknown primitive fails closed: a missing row is not a licence to assume the
// barrier orders the call.
constexpr bool UbCopyPrimitiveIsPipeModeled(UbHandoffCopyPrimitive primitive)
{
    for (std::size_t i = 0; i < sizeof(kUbCopyPrimitiveFacts) / sizeof(kUbCopyPrimitiveFacts[0]); ++i) {
        if (kUbCopyPrimitiveFacts[i].primitive == primitive)
            return kUbCopyPrimitiveFacts[i].caller_annotation == UbPipeAnnotation::V;
    }
    return false;
}

constexpr bool UbCopyPrimitiveMaskIsElementCount(UbHandoffCopyPrimitive primitive)
{
    for (std::size_t i = 0; i < sizeof(kUbCopyPrimitiveFacts) / sizeof(kUbCopyPrimitiveFacts[0]); ++i) {
        if (kUbCopyPrimitiveFacts[i].primitive == primitive)
            return kUbCopyPrimitiveFacts[i].mask_semantics == UbCopyMaskSemantics::ElementCount;
    }
    return false;
}

static_assert(UbCopyPrimitiveMaskIsElementCount(UbHandoffCopyPrimitive::VCopy),
              "the V-annotated handoff copy takes an element count, not a bitmask");

// `CopyRepeatParams` is `(dstStride, srcStride, dstRepeatSize, srcRepeatSize)`.
// The authoritative contiguous-copy defaults are the CANN C-API ones
// (`asc/impl/c_api/instr_impl/npu_arch_2201/utils_impl/utils_impl.h:42-54`):
// `ASC_C_API_DEFAULT_BLOCK_STRIDE = 1` and `ASC_C_API_DEFAULT_REPEAT_STRIDE = 8`,
// and the CATLASS reference uses the same pair
// (`catlass/epilogue/tile/tile_broadcast_inplace_by_row.hpp:44-48`). `dstStride`
// is the intra-repeat datablock stride, so `0` collapses every datablock onto
// one address and silently corrupts the copy; the repeat size is the repeat
// span in datablocks (`BLK_NUM_PER_VECTOR_FRACTAL = 8`). This is the second
// implicit rule the `Copy` call site depends on, so it is ledger data too.
enum class UbCopyRepeatField : std::uint8_t {
    DstStride,
    SrcStride,
    DstRepeatSize,
    SrcRepeatSize,
};

struct UbCopyRepeatFieldFact {
    UbCopyRepeatField field;
    std::uint16_t contiguous_value;
    const char* unit;
    const char* source_ref;
};

constexpr UbCopyRepeatFieldFact kUbCopyRepeatFieldFacts[] = {
    {UbCopyRepeatField::DstStride, 1, "datablock",
     "asc/impl/c_api/instr_impl/npu_arch_2201/utils_impl/utils_impl.h:46-49"},
    {UbCopyRepeatField::SrcStride, 1, "datablock",
     "asc/impl/c_api/instr_impl/npu_arch_2201/utils_impl/utils_impl.h:46-49"},
    {UbCopyRepeatField::DstRepeatSize, 8, "datablock",
     "asc/impl/c_api/instr_impl/npu_arch_2201/utils_impl/utils_impl.h:51-54"},
    {UbCopyRepeatField::SrcRepeatSize, 8, "datablock",
     "asc/impl/c_api/instr_impl/npu_arch_2201/utils_impl/utils_impl.h:51-54"},
};

// Unknown field fails closed: a missing row is not a licence to assume the
// contiguous value.
constexpr std::uint16_t UbCopyRepeatFieldValue(UbCopyRepeatField field)
{
    for (std::size_t i = 0; i < sizeof(kUbCopyRepeatFieldFacts) / sizeof(kUbCopyRepeatFieldFacts[0]); ++i) {
        if (kUbCopyRepeatFieldFacts[i].field == field)
            return kUbCopyRepeatFieldFacts[i].contiguous_value;
    }
    return 0;
}

constexpr std::uint16_t UbCopyContiguousBlockStride()
{
    return UbCopyRepeatFieldValue(UbCopyRepeatField::DstStride);
}

constexpr std::uint16_t UbCopyContiguousRepeatStride()
{
    return UbCopyRepeatFieldValue(UbCopyRepeatField::DstRepeatSize);
}

static_assert(UbCopyContiguousBlockStride() != 0,
              "a zero datablock stride collapses every datablock onto one address and corrupts the copy");
static_assert(UbCopyContiguousRepeatStride() == 8, "the contiguous repeat span is one vector fractal");

enum class UbHandoffOrdering : std::uint8_t {
    PipeV,
    PipeAll,
};

struct UbHandoffCopyContract {
    UbHandoffCopyPrimitive primitive;
    UbHandoffOrdering ordering;
    bool consumer_is_v_pipe;
};

constexpr bool ValidateUbHandoffCopyContract(const UbHandoffCopyContract& contract)
{
    if (!contract.consumer_is_v_pipe || contract.ordering == UbHandoffOrdering::PipeAll)
        return true;
    return UbCopyPrimitiveIsPipeModeled(contract.primitive);
}

constexpr UbHandoffCopyContract kStreamHandoffCopy = {UbHandoffCopyPrimitive::VCopy, UbHandoffOrdering::PipeV, true};
static_assert(ValidateUbHandoffCopyContract(kStreamHandoffCopy));
static_assert(!ValidateUbHandoffCopyContract({UbHandoffCopyPrimitive::DataCopyUbToUb, UbHandoffOrdering::PipeV, true}),
              "an unannotated UB->UB DataCopy is not ordered by PipeBarrier<PIPE_V>");
static_assert(!ValidateUbHandoffCopyContract({UbHandoffCopyPrimitive::VMuls, UbHandoffOrdering::PipeV, true}),
              "Muls has no caller-side __inout_pipe__ on this CANN, so PipeBarrier<PIPE_V> does not order it");
static_assert(!ValidateUbHandoffCopyContract({UbHandoffCopyPrimitive::VAdds, UbHandoffOrdering::PipeV, true}),
              "Adds has no caller-side __inout_pipe__ on this CANN, so PipeBarrier<PIPE_V> does not order it");
static_assert(ValidateUbHandoffCopyContract({UbHandoffCopyPrimitive::DataCopyUbToUb, UbHandoffOrdering::PipeAll, true}),
              "PIPE_ALL does order the unannotated UB->UB DataCopy");
static_assert(ValidateUbHandoffCopyContract({UbHandoffCopyPrimitive::DataCopyUbToUb, UbHandoffOrdering::PipeV, false}),
              "a non-V consumer does not need the V-pipe ordering");

// The online-softmax epilogue keeps its per-row recurrence state in UB and
// advances it with UB->UB copies that sit on the same V-pipe critical path as
// the stream handoff:
//   `hm = lm` -> `gm = hm`            (immediate V->V edge, same stack)
//   `gm = hm` -> next stack's `dm = gm - hm` and `gl = dm*gl`
//   `gl = ll` -> next stack's `gl = dm*gl` and the task-final snapshot
// All three are task-boundary state copies, so an unannotated `DataCopy` there
// escapes `PipeBarrier<PIPE_V>` exactly like the handoff did; the anchor's
// `PIPE_ALL` drains were the only thing ordering them. They therefore need the
// same V-annotated primitive as `kStreamHandoffCopy`.
constexpr UbHandoffCopyContract kSoftmaxStateCopy = {UbHandoffCopyPrimitive::VCopy, UbHandoffOrdering::PipeV, true};
static_assert(ValidateUbHandoffCopyContract(kSoftmaxStateCopy));
static_assert(!ValidateUbHandoffCopyContract({UbHandoffCopyPrimitive::DataCopyUbToUb, UbHandoffOrdering::PipeV, true}),
              "the online-softmax recurrence-state copies must not use the unannotated UB->UB DataCopy");

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

constexpr bool ValidateDiagnosticStageCut(const DiagnosticStageCutContract& contract)
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

enum class LayoutOwnerRole : std::uint8_t { Producer, Consumer };
enum class AttentionPayloadRole : std::uint8_t { P, V };

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

constexpr bool MultiplyWouldOverflow(std::size_t lhs, std::size_t rhs) { return lhs != 0U && rhs > kMaxSize / lhs; }

constexpr std::size_t DensePartBytes(std::size_t rows, std::size_t cols, std::size_t element_bytes)
{
    if (rows == 0U || cols == 0U || element_bytes == 0U || MultiplyWouldOverflow(rows, cols))
        return kMaxSize;
    const std::size_t elements = rows * cols;
    return MultiplyWouldOverflow(elements, element_bytes) ? kMaxSize : elements * element_bytes;
}

constexpr bool ValidateTypedPartLayout(const TypedPartLayout& layout)
{
    const std::size_t part_bytes = DensePartBytes(layout.rows, layout.cols, layout.element_bytes);
    return part_bytes != kMaxSize && layout.part_count > 0U && layout.part_stride_bytes == part_bytes &&
           !MultiplyWouldOverflow(layout.part_count, layout.part_stride_bytes) &&
           layout.owner_bytes == layout.part_count * layout.part_stride_bytes;
}

constexpr bool ValidateTypedLayoutHandoff(const TypedPartLayout& producer, const TypedPartLayout& consumer)
{
    return producer.owner == LayoutOwnerRole::Producer && consumer.owner == LayoutOwnerRole::Consumer &&
           producer.payload == consumer.payload && producer.rows == consumer.rows && producer.cols == consumer.cols &&
           producer.element_bytes == consumer.element_bytes && producer.part_count == consumer.part_count &&
           producer.part_stride_bytes == consumer.part_stride_bytes && producer.owner_bytes == consumer.owner_bytes &&
           ValidateTypedPartLayout(producer) && ValidateTypedPartLayout(consumer);
}

constexpr std::size_t TypedPartOffset(const TypedPartLayout& layout, std::size_t part)
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

struct RowReductionTreeContract {
    std::size_t columns;
    std::size_t vector_elements;
    std::size_t block_elements;
    std::size_t rows_per_loop;
    std::size_t score_capacity_elements;
    std::size_t first_partial_capacity_elements;
    std::size_t second_partial_capacity_elements;
};

constexpr bool ValidateD32RowReduction384(const RowReductionTreeContract& contract)
{
    if (contract.columns != 384U || contract.vector_elements != 64U || contract.block_elements != 8U ||
        contract.rows_per_loop == 0U || contract.columns % contract.vector_elements != 0U) {
        return false;
    }
    const std::size_t segments = contract.columns / contract.vector_elements;
    const std::size_t first_partials = contract.rows_per_loop * contract.columns / contract.block_elements;
    const std::size_t second_partials = contract.rows_per_loop * contract.block_elements;
    return segments == 6U && contract.rows_per_loop * contract.columns <= contract.score_capacity_elements &&
           first_partials <= contract.first_partial_capacity_elements &&
           second_partials <= contract.second_partial_capacity_elements;
}

constexpr RowReductionTreeContract kD32RowReduction384 = {384U, 64U, 8U, 16U, 8192U, 1024U, 1024U};
static_assert(ValidateD32RowReduction384(kD32RowReduction384));
static_assert(!ValidateD32RowReduction384({384U, 64U, 8U, 24U, 8192U, 1024U, 1024U}),
              "24 rows overflow the first 384-column reduction scratch owner");
static_assert(!ValidateD32RowReduction384({320U, 64U, 8U, 16U, 8192U, 1024U, 1024U}),
              "the specialized reduction tree owns exactly six 64-column segments");

struct Bf16D16RowReduction384GuardContract {
    std::size_t dim;
    std::size_t tile_numel;
    std::size_t columns;
    std::size_t vector_elements;
    std::size_t block_elements;
    std::size_t rows_per_loop;
    std::size_t score_capacity_elements;
    std::size_t first_partial_capacity_elements;
    std::size_t second_partial_capacity_elements;
    bool requires_text;
    bool requires_pair_axis;
};

constexpr bool ValidateBf16D16RowReduction384(const Bf16D16RowReduction384GuardContract& contract)
{
    return contract.dim == 16U && contract.tile_numel == 384U && contract.columns == 384U &&
           contract.vector_elements == 64U && contract.block_elements == 8U && contract.rows_per_loop == 16U &&
           contract.score_capacity_elements == 8192U && contract.first_partial_capacity_elements == 1024U &&
           contract.second_partial_capacity_elements == 1024U && contract.requires_text &&
           contract.requires_pair_axis &&
           ValidateD32RowReduction384({contract.columns, contract.vector_elements, contract.block_elements,
                                       contract.rows_per_loop, contract.score_capacity_elements,
                                       contract.first_partial_capacity_elements,
                                       contract.second_partial_capacity_elements});
}

constexpr Bf16D16RowReduction384GuardContract kBf16D16RowReduction384 = {
    16U, 384U, 384U, 64U, 8U, 16U, 8192U, 1024U, 1024U, true, true};
static_assert(ValidateBf16D16RowReduction384(kBf16D16RowReduction384));
static_assert(!ValidateBf16D16RowReduction384(
                  {16U, 384U, 384U, 64U, 8U, 16U, 8192U, 1024U, 1024U, false, true}),
              "the BF16 D16 specialization must remain restricted to text-pair metadata");
static_assert(!ValidateBf16D16RowReduction384(
                  {16U, 384U, 384U, 64U, 8U, 24U, 8192U, 1024U, 1024U, true, true}),
              "the BF16 D16 reduction tree cannot exceed its 16-row scratch contract");

struct RowReductionDtypeContract {
    std::size_t input_element_bytes;
    std::size_t score_element_bytes;
    std::size_t state_element_bytes;
    std::size_t probability_element_bytes;
    std::size_t output_element_bytes;
    bool probability_uses_cast_none;
};

constexpr bool ValidateBf16D32RowReduction384Dtype(const RowReductionDtypeContract& contract)
{
    return contract.input_element_bytes == 2U && contract.score_element_bytes == 4U &&
           contract.state_element_bytes == 4U && contract.probability_element_bytes == 2U &&
           contract.output_element_bytes == 2U && contract.probability_uses_cast_none;
}

constexpr RowReductionDtypeContract kBf16D32RowReduction384Dtype = {2U, 4U, 4U, 2U, 2U, true};
static_assert(ValidateBf16D32RowReduction384Dtype(kBf16D32RowReduction384Dtype));
static_assert(!ValidateBf16D32RowReduction384Dtype({2U, 2U, 4U, 2U, 2U, true}),
              "BF16 D32 row reduction must retain FP32 score ownership");
static_assert(!ValidateBf16D32RowReduction384Dtype({2U, 4U, 4U, 2U, 2U, false}),
              "BF16 outer output still consumes the established FP16 probability publication");

struct D32TextRowReduction384Contract {
    std::size_t input_element_bytes;
    std::size_t score_element_bytes;
    std::size_t state_element_bytes;
    std::size_t probability_element_bytes;
    std::size_t tile_elements;
    std::size_t specialized_columns;
    bool exact_column_guard;
    bool non_specialized_tail_fallback;
};

constexpr bool ValidateD32TextRowReduction384(const D32TextRowReduction384Contract& contract)
{
    return contract.input_element_bytes == 2U && contract.score_element_bytes == 4U &&
           contract.state_element_bytes == 4U && contract.probability_element_bytes == 2U &&
           contract.tile_elements == 384U && contract.specialized_columns == 384U && contract.exact_column_guard &&
           contract.non_specialized_tail_fallback;
}

constexpr D32TextRowReduction384Contract kD32TextRowReduction384 = {
    2U, 4U, 4U, 2U, 384U, 384U, true, true};
static_assert(ValidateD32TextRowReduction384(kD32TextRowReduction384));
static_assert(!ValidateD32TextRowReduction384({2U, 4U, 4U, 2U, 384U, 384U, false, true}),
              "the text-capable reduction requires an exact 384-column runtime guard");
static_assert(!ValidateD32TextRowReduction384({2U, 4U, 4U, 2U, 384U, 384U, true, false}),
              "non-384 text tails must retain the generic row-reduction path");

struct Bf16D32SingleQueryPackedWindowContract {
    std::size_t query_count;
    std::size_t query_rows;
    std::size_t dim;
    std::size_t tile_elements;
    std::size_t segment_count;
    std::size_t segment_rows;
    std::size_t block_rows;
    std::size_t max_blocks_per_stack;
    std::size_t ring_slots;
    std::size_t prelaunch_depth;
    std::size_t score_element_bytes;
    std::size_t probability_element_bytes;
    std::size_t otmp_element_bytes;
    bool explicit_block_offsets;
    bool token_order_preserved;
    bool qk_pv_term_count_preserved;
    bool independent_query_owner;
};

constexpr std::size_t SegmentedStackCount(const Bf16D32SingleQueryPackedWindowContract& contract)
{
    const std::size_t blocks_per_segment = contract.segment_rows / contract.block_rows;
    return contract.segment_count *
           ((blocks_per_segment + contract.max_blocks_per_stack - 1U) / contract.max_blocks_per_stack);
}

constexpr std::size_t PackedStackCount(const Bf16D32SingleQueryPackedWindowContract& contract)
{
    const std::size_t total_blocks = contract.segment_count * contract.segment_rows / contract.block_rows;
    return (total_blocks + contract.max_blocks_per_stack - 1U) / contract.max_blocks_per_stack;
}

constexpr std::size_t OtmpPublicationBytes(const Bf16D32SingleQueryPackedWindowContract& contract,
                                           std::size_t stack_count)
{
    return stack_count * contract.query_rows * contract.dim * contract.otmp_element_bytes;
}

constexpr bool ValidateBf16D32SingleQueryPackedWindow(
    const Bf16D32SingleQueryPackedWindowContract& contract)
{
    if (contract.query_count != 1U || contract.query_rows != 128U || contract.dim != 32U ||
        contract.tile_elements != 384U || contract.segment_count != 3U || contract.segment_rows != 1152U ||
        contract.block_rows != 128U || contract.max_blocks_per_stack != 4U || contract.ring_slots != 5U ||
        contract.prelaunch_depth != 4U || contract.score_element_bytes != 4U ||
        contract.probability_element_bytes != 2U || contract.otmp_element_bytes != 4U ||
        contract.segment_rows % contract.block_rows != 0U) {
        return false;
    }
    return contract.explicit_block_offsets && contract.token_order_preserved &&
           contract.qk_pv_term_count_preserved && contract.independent_query_owner &&
           contract.ring_slots >= contract.prelaunch_depth + 1U && SegmentedStackCount(contract) == 9U &&
           PackedStackCount(contract) == 7U &&
           OtmpPublicationBytes(contract, SegmentedStackCount(contract)) == 147456U &&
           OtmpPublicationBytes(contract, PackedStackCount(contract)) == 114688U;
}

constexpr Bf16D32SingleQueryPackedWindowContract kBf16D32SingleQueryPackedWindow = {
    1U, 128U, 32U, 384U, 3U, 1152U, 128U, 4U, 5U, 4U, 4U, 2U, 4U, true, true, true, true};
static_assert(ValidateBf16D32SingleQueryPackedWindow(kBf16D32SingleQueryPackedWindow));
static_assert(!ValidateBf16D32SingleQueryPackedWindow(
                  {1U, 128U, 32U, 384U, 3U, 1152U, 128U, 4U, 5U, 4U, 4U, 2U, 4U, true, true, true, false}),
              "packed D32 must not merge independent query owners");
static_assert(!ValidateBf16D32SingleQueryPackedWindow(
                  {1U, 128U, 32U, 384U, 3U, 1152U, 128U, 3U, 5U, 4U, 4U, 2U, 4U, true, true, true, true}),
              "the packed receipt requires the four-block source-owned QK/PV stack");

// The D32 packed route may index its five-slot score/P/OTmp ring with a continuous
// per-core stack sequence instead of resetting at every 128-row query-chunk task.
// That is only legal while the ring stays strictly deeper than the publication
// delay on every task boundary, and while each slot still carries the owning task's
// local stack index, stack total, and output base for the first/last-stack
// recurrence flags. Payload extent, event pairing, and allocations must not change.
struct D32PackedStreamSlotContract {
    std::size_t ring_slots;
    std::size_t prelaunch_depth;
    std::size_t query_rows;
    std::size_t dim;
    std::size_t max_blocks_per_stack;
    std::size_t block_rows;
    bool per_task_first_last_retained;
    bool slot_index_continuous_across_tasks;
    bool payload_extent_unchanged;
    bool consumer_release_precedes_next_writer;
};

constexpr bool ValidateD32PackedStreamSlot(const D32PackedStreamSlotContract& contract)
{
    if (contract.ring_slots == 0U || contract.prelaunch_depth == 0U || contract.query_rows != 128U ||
        contract.dim != 32U || contract.max_blocks_per_stack != 4U || contract.block_rows != 128U) {
        return false;
    }
    if (contract.ring_slots <= contract.prelaunch_depth) {
        return false;
    }
    return contract.per_task_first_last_retained && contract.slot_index_continuous_across_tasks &&
           contract.payload_extent_unchanged && contract.consumer_release_precedes_next_writer;
}

constexpr D32PackedStreamSlotContract kD32PackedStreamSlot = {
    5U, 4U, 128U, 32U, 4U, 128U, true, true, true, true};
static_assert(ValidateD32PackedStreamSlot(kD32PackedStreamSlot));
static_assert(!ValidateD32PackedStreamSlot({4U, 4U, 128U, 32U, 4U, 128U, true, true, true, true}),
              "a four-slot D32 ring cannot exceed its own publication delay across task boundaries");
static_assert(!ValidateD32PackedStreamSlot({5U, 4U, 128U, 32U, 4U, 128U, false, true, true, true}),
              "the continuous D32 stream must retain per-task first/last stack ownership");
static_assert(!ValidateD32PackedStreamSlot({5U, 4U, 128U, 32U, 4U, 128U, true, false, true, true}),
              "the continuous D32 stream is defined by a slot index that does not reset per task");
static_assert(!ValidateD32PackedStreamSlot({5U, 4U, 128U, 32U, 4U, 128U, true, true, true, false}),
              "a slot may not be rewritten before its delayed consumer released it");

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

constexpr bool ValidateDualProbabilityNumericContract(const DualProbabilityNumericContract& contract)
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

// D16 narrow-score bridge. The D16 no-text pair route publishes its 128x512 fp32
// score tile through the public fixpipe fp16 cast mode and upcasts one staged row
// tile per row-loop iteration, so both directions of the score GM round trip halve
// while every other dtype, layout, event and accumulator stays as it is. The score
// workspace keeps its fp32 sizing, so the half view has twice as many elements for
// the same bytes: an fp32 stage-base offset k maps to the half element offset 2k,
// while offsets inside one stage stay unscaled because the half tile keeps the fp32
// element geometry and occupies half the bytes.
constexpr std::size_t kD16HalfScoreElemBytes = 2U;
constexpr std::size_t kD16Fp32ScoreElemBytes = 4U;
constexpr std::size_t kD16NarrowScoreStageBytes = 16U * 1024U;
constexpr std::size_t kD16NarrowScoreStageElems = kD16NarrowScoreStageBytes / kD16HalfScoreElemBytes;

// The staged owner must lie between two real owners, fit inside UB, and hold the
// largest narrow row tile the compact row loop can produce.
constexpr bool ValidateD16NarrowScoreStage(std::size_t stage_offset_bytes, std::size_t stage_bytes,
                                           std::size_t previous_owner_end_bytes, std::size_t next_owner_offset_bytes,
                                           std::size_t ub_bytes, std::size_t row_tile_elems,
                                           std::size_t stage_capacity_elems)
{
    return stage_offset_bytes >= previous_owner_end_bytes &&
           stage_offset_bytes + stage_bytes <= next_owner_offset_bytes && next_owner_offset_bytes <= ub_bytes &&
           row_tile_elems > 0U && stage_capacity_elems >= row_tile_elems;
}

// Real geometry: the stage is the free 16 KiB region between the rescale `go` owner
// (one 4096-float tile after its base) and the softmax TV owner, and it holds the
// widest narrow row tile (16 rows x 512 halves = MAX_UB_S_ELEM_NUM halves).
static_assert(ValidateD16NarrowScoreStage(147456U, kD16NarrowScoreStageBytes, 131072U + 16384U, 163840U, 196608U,
                                          8192U, kD16NarrowScoreStageElems));
static_assert(!ValidateD16NarrowScoreStage(131072U, kD16NarrowScoreStageBytes, 131072U + 16384U, 163840U, 196608U,
                                           8192U, kD16NarrowScoreStageElems),
              "the narrow-score stage must start at or after the previous owner's end");
static_assert(!ValidateD16NarrowScoreStage(147456U, 32U * 1024U, 131072U + 16384U, 163840U, 196608U, 8192U,
                                           kD16NarrowScoreStageElems),
              "the narrow-score stage must not overrun the next owner");
static_assert(!ValidateD16NarrowScoreStage(147456U, kD16NarrowScoreStageBytes, 131072U + 16384U, 163840U, 196608U,
                                           16384U, kD16NarrowScoreStageElems),
              "the narrow-score row tile must fit the staged owner");
static_assert(kD16HalfScoreElemBytes * 2U == kD16Fp32ScoreElemBytes,
              "the narrow score element must be exactly half of the fp32 element");

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

template <const auto& Regions>
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
enum class Access : std::uint8_t { Read, Write };
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
    constexpr bool ok() const { return error == Error::Ok; }
};
struct RuntimeOwner {
    std::uintptr_t base;
    std::size_t bytes;
    std::size_t alignment;
};
constexpr std::size_t kInvalidSize = static_cast<std::size_t>(-1);
constexpr bool AddWouldOverflow(std::size_t a, std::size_t b) { return b > kInvalidSize - a; }
constexpr bool MultiplyWouldOverflow(std::size_t a, std::size_t b) { return a != 0U && b > kInvalidSize / a; }
constexpr std::size_t Mul(std::size_t a, std::size_t b) { return MultiplyWouldOverflow(a, b) ? kInvalidSize : a * b; }
constexpr std::size_t Add(std::size_t a, std::size_t b) { return AddWouldOverflow(a, b) ? kInvalidSize : a + b; }
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
constexpr Result ValidateAccess(const RuntimeOwner& owner, std::uintptr_t address, std::size_t span, Access access,
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
inline Result ValidateLinearLaunch(const RuntimeOwner& out, std::uintptr_t outp, const RuntimeOwner& in,
                                   std::uintptr_t inp, const RuntimeOwner& w, std::uintptr_t wp, std::uint32_t rows,
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
    Result r = ValidateAccess(in, inp + Mul(Add(Mul(ro, K), ko), E), Span(TR, TK, K, E), Access::Read, task - 1U,
                              ki - 1U);
    if (!r.ok())
        return r;
    r = ValidateAccess(w, wp + Mul(Add(Mul(no, K), ko), E), Span(TN, TK, K, E), Access::Read, task - 1U, ki - 1U);
    if (!r.ok())
        return r;
    return ValidateAccess(out, outp + Mul(Add(Mul(ro, N), no), E), Span(TR, TN, N, E), Access::Write, task - 1U, ki);
}
} // namespace opforge::ascend_c::resource_ledger::gm

// ---------------------------------------------------------------------------
// Solution-local extension: D32 packed-stream UB region table.
//
// The base ledger above carries the region/conflict machinery but the D32
// packed-stream kernel previously declared its UB offsets as raw constants with
// no ledger binding, so a region overlap could not be caught at compile time.
// This table binds every D32 packed-stream UB owner to its real allocation
// offset, byte extent, and lifetime. `RegionsConflict` fails closed when two
// regions on the same storage overlap in both address and lifetime.
//
// Offsets are the exact kernel constants:
//   UB_REGION_BYTES = 16 * 1024, UB_SUBREGION_BYTES = 1024
//   LS  [0, 65536)          softmax score, 2 x 8192 floats (ping-pong)
//   LP  [65536, 98304)      softmax P, 2 x 8192 half
//   LO  [98304, 131072)     rescale local / softmax center
//   GO  [131072, 163840)    rescale output
//   TV  [163840, 168960)    softmax reduce scratch (16-row loop: first partials
//                           [163840, 166912), second partials [167936, 168448))
//   PARK [168960, 169984)   stream row-sum park slot
//   SNAP [169984, 172032)   stream row-sum snapshot, 8 x 64 floats
//   LM  [172032, 173056)    softmax local max
//   HM  [173056, 174080)    softmax global max
//   GM  [174080, 175104)    softmax global max (running)
//   LL  [175104, 176128)    softmax local sum
//   GL  [176128, 177152)    softmax global sum (running row sum)
//   DM  [177152, 179712)    softmax per-slot dm, 5 x 128 floats
//   MASK[180224, 196608)    softmax mask
// ---------------------------------------------------------------------------
namespace opforge::ascend_c::resource_ledger::d32_packed_stream {

constexpr std::uint64_t kUbStorage = 1;
constexpr std::size_t kUbCapacityBytes = 196608;  // 192 KiB
constexpr std::size_t kDataBlockBytes = 32;

constexpr std::size_t kLsOffset = 0;
constexpr std::size_t kLsBytes = 65536;
constexpr std::size_t kLpOffset = 65536;
constexpr std::size_t kLpBytes = 32768;
constexpr std::size_t kLoOffset = 98304;
constexpr std::size_t kLoBytes = 32768;
constexpr std::size_t kGoOffset = 131072;
constexpr std::size_t kGoBytes = 32768;
// The TV region is the whole 8 KiB region-10 slot, but the reduce only writes
// the first-partial block [163840, 166912) and the second-partial block
// [167936, 168448). The park/snapshot slots live at [168960, 172032), i.e.
// inside the *allocated* region but outside the *live* extent. The ledger
// therefore models the live extent, and the two partial blocks are declared
// separately so the gap between them is not silently claimed.
constexpr std::size_t kTvOffset = 163840;
constexpr std::size_t kTvFirstPartialBytes = 3072;
constexpr std::size_t kTvSecondPartialOffset = 167936;
constexpr std::size_t kTvSecondPartialBytes = 512;
constexpr std::size_t kParkOffset = 168960;
constexpr std::size_t kParkBytes = 1024;
constexpr std::size_t kSnapshotOffset = 169984;
constexpr std::size_t kSnapshotBytes = 2048;
constexpr std::size_t kLmOffset = 172032;
constexpr std::size_t kLmBytes = 1024;
constexpr std::size_t kHmOffset = 173056;
constexpr std::size_t kHmBytes = 1024;
constexpr std::size_t kGmOffset = 174080;
constexpr std::size_t kGmBytes = 1024;
constexpr std::size_t kLlOffset = 175104;
constexpr std::size_t kLlBytes = 1024;
constexpr std::size_t kGlOffset = 176128;
constexpr std::size_t kGlBytes = 1024;
constexpr std::size_t kDmOffset = 177152;
constexpr std::size_t kDmBytes = 2560;
constexpr std::size_t kMaskOffset = 180224;
constexpr std::size_t kMaskBytes = 16384;

// Lifetime ids. 0 = per-stream-position scratch, 1 = stream-persistent state.
// Regions with different lifetimes may share an address; regions with the same
// lifetime must be disjoint.
constexpr std::uint32_t kPerPosition = 0;
constexpr std::uint32_t kPersistent = 1;

constexpr Region kD32PackedStreamUbRegions[] = {
    {kUbStorage, kLsOffset, kLsBytes, kDataBlockBytes, kPerPosition, kPerPosition + 1, kUbCapacityBytes},
    {kUbStorage, kLpOffset, kLpBytes, kDataBlockBytes, kPerPosition, kPerPosition + 1, kUbCapacityBytes},
    {kUbStorage, kLoOffset, kLoBytes, kDataBlockBytes, kPerPosition, kPerPosition + 1, kUbCapacityBytes},
    {kUbStorage, kGoOffset, kGoBytes, kDataBlockBytes, kPerPosition, kPerPosition + 1, kUbCapacityBytes},
    {kUbStorage, kTvOffset, kTvFirstPartialBytes, kDataBlockBytes, kPerPosition, kPerPosition + 1, kUbCapacityBytes},
    {kUbStorage, kTvSecondPartialOffset, kTvSecondPartialBytes, kDataBlockBytes, kPerPosition, kPerPosition + 1,
     kUbCapacityBytes},
    {kUbStorage, kParkOffset, kParkBytes, kDataBlockBytes, kPerPosition, kPerPosition + 1, kUbCapacityBytes},
    {kUbStorage, kSnapshotOffset, kSnapshotBytes, kDataBlockBytes, kPerPosition, kPerPosition + 1, kUbCapacityBytes},
    {kUbStorage, kLmOffset, kLmBytes, kDataBlockBytes, kPerPosition, kPerPosition + 1, kUbCapacityBytes},
    {kUbStorage, kHmOffset, kHmBytes, kDataBlockBytes, kPerPosition, kPerPosition + 1, kUbCapacityBytes},
    {kUbStorage, kGmOffset, kGmBytes, kDataBlockBytes, kPerPosition, kPerPosition + 1, kUbCapacityBytes},
    {kUbStorage, kLlOffset, kLlBytes, kDataBlockBytes, kPerPosition, kPerPosition + 1, kUbCapacityBytes},
    {kUbStorage, kGlOffset, kGlBytes, kDataBlockBytes, kPersistent, kPersistent + 1, kUbCapacityBytes},
    {kUbStorage, kDmOffset, kDmBytes, kDataBlockBytes, kPersistent, kPersistent + 1, kUbCapacityBytes},
    {kUbStorage, kMaskOffset, kMaskBytes, kDataBlockBytes, kPerPosition, kPerPosition + 1, kUbCapacityBytes},
};

using D32PackedStreamUbLedger = StaticLedger<kD32PackedStreamUbRegions>;
static_assert(D32PackedStreamUbLedger::valid,
              "D32 packed-stream UB regions must be disjoint within their lifetime and inside capacity");

// Negative controls: the same table with a deliberately overlapping pair must
// fail, so the predicate is proven to reject the failure mode it exists for.
constexpr Region kD32PackedStreamUbOverlap[] = {
    {kUbStorage, kTvOffset, kTvFirstPartialBytes, kDataBlockBytes, kPerPosition, kPerPosition + 1, kUbCapacityBytes},
    {kUbStorage, kSnapshotOffset, kSnapshotBytes, kDataBlockBytes, kPerPosition, kPerPosition + 1, kUbCapacityBytes},
    {kUbStorage, kTvOffset + 1024, 4096, kDataBlockBytes, kPerPosition, kPerPosition + 1, kUbCapacityBytes},
};
static_assert(!StaticLedger<kD32PackedStreamUbOverlap>::valid,
              "the D32 UB ledger must reject a per-position region overlapping the reduce scratch");

constexpr Region kD32PackedStreamUbOutOfCapacity[] = {
    {kUbStorage, kUbCapacityBytes - 1024, 4096, kDataBlockBytes, kPerPosition, kPerPosition + 1, kUbCapacityBytes},
};
static_assert(!StaticLedger<kD32PackedStreamUbOutOfCapacity>::valid,
              "the D32 UB ledger must reject a region past the UB capacity");

} // namespace opforge::ascend_c::resource_ledger::d32_packed_stream

#endif // OPFORGE_RESOURCE_LEDGER_HOST_PROJECTION_ONLY
