/**
 * \file profiler_core.h
 * \brief Device-side profiling mechanism — GM ring buffer write + DCCI flush.
 *
 * Contains buffer layout constants, GmWrite helpers, Profiler::Init/Mark,
 * and PROF_INIT/PROF_MARK macros. NO event IDs — those belong in your
 * operator-specific profiler.h alongside this file.
 *
 * Include chain: profiler.h → profiler_core.h → profiler_config.h
 *
 * When ENABLE_PROFILING is not defined, PROF_INIT and PROF_MARK expand
 * to ((void)0) — zero overhead.
 *
 * === GM Access Pattern ===
 * All GM reads/writes use direct __gm__ pointer dereference (same as hcomm's
 * cacheWriteThrough and asc-devkit's FlushDataCache). GlobalTensor is used
 * solely as the vehicle for DataCacheCleanAndInvalid (DCCI).
 *
 * Version: 1.0.0
 */

#pragma once

#include "kernel_basic_intf.h"

#ifdef ENABLE_PROFILING

#include "profiler_config.h"

static_assert(PROF_N_CORES > 0, "PROF_N_CORES must be > 0. Check profiler_config.h");

// Guard: EVT_OVERFLOW (0xFFFF) is reserved as the ring buffer overflow sentinel.
// User event IDs in profiler.h must not use this value.

namespace Profiler {

__aicore__ inline uint32_t GetCoreId() {
    uint32_t id = AscendC::GetBlockIdx();
    // AIC cores are numbered after AIV cores in the profiler ring buffer.
    // PROF_AIV_CORES = 0 means AIV-only mode (no AIC cores).
#if PROF_AIV_CORES > 0
  #ifdef ASCEND_IS_AIC
    if ASCEND_IS_AIC {
        id += PROF_AIV_CORES;
    }
  #endif
  // If ASCEND_IS_AIC is not defined (AIV-only kernel), all cores use GetBlockIdx() directly.
#endif
    return id;
}

// Write a 32-bit value to GM via direct pointer, then DCCI flush for host visibility.
// Same pattern as hcomm cacheWriteThrough / asc-devkit FlushDataCache.
// WARNING: the DCCI flush on the next line is MANDATORY — copying only `*addr = value`
// without it will NOT reach HBM (host D2H reads stale/zero). Do not rely on DCCI
// side-effects from other write sites; each GM write must flush its own cache line.
__aicore__ inline void GmWrite32(__gm__ uint32_t* addr, uint32_t value) {
    *addr = value;
    AscendC::GlobalTensor<uint32_t> gt;
    gt.SetGlobalBuffer(addr, 1);
    AscendC::DataCacheCleanAndInvalid<uint32_t, AscendC::CacheLine::SINGLE_CACHE_LINE, AscendC::DcciDst::CACHELINE_OUT>(gt);
}

// Write a 64-bit value to GM via direct pointer, single DCCI flush.
__aicore__ inline void GmWrite64(__gm__ uint32_t* addr, uint64_t value) {
    addr[0] = static_cast<uint32_t>(value);
    addr[1] = static_cast<uint32_t>(value >> 32);
    AscendC::GlobalTensor<uint32_t> gt;
    gt.SetGlobalBuffer(addr, 2);
    AscendC::DataCacheCleanAndInvalid<uint32_t, AscendC::CacheLine::SINGLE_CACHE_LINE, AscendC::DcciDst::CACHELINE_OUT>(gt);
}

__aicore__ inline void Init(__gm__ uint8_t* profBufGm) {
    if (profBufGm == (__gm__ uint8_t*)0) return;
    uint32_t coreId = GetCoreId();
    if (coreId >= PROF_N_CORES) return;
    // WARNING: If PROF_N_CORES is set too low at compile time, cores with
    // ID >= PROF_N_CORES silently skip profiling. No runtime error is raised.
    // Verify PROF_N_CORES matches your kernel's KERNEL_TASK_TYPE_DEFAULT.
    __gm__ uint32_t* ringIdxPtr = (__gm__ uint32_t*)(profBufGm + coreId * PROF_RINGIDX_ALIGN);
    GmWrite32(ringIdxPtr, 0);
}

__aicore__ inline void Mark(__gm__ uint8_t* profBufGm, uint32_t eventId, uint32_t payload = 0) {
    if (profBufGm == (__gm__ uint8_t*)0) return;
    uint32_t coreId = GetCoreId();
    if (coreId >= PROF_N_CORES) return;
    // Prevent user events from colliding with the overflow sentinel
    if (eventId == EVT_OVERFLOW) return;
    __gm__ uint32_t* ringIdxPtr = (__gm__ uint32_t*)(profBufGm + coreId * PROF_RINGIDX_ALIGN);

    // This read relies on the previous GmWrite32's DCCI having invalidated
    // this cache line. Ascend C guarantees sequential execution within a core
    // and no reordering across DCCI barriers.
    uint32_t idx = *ringIdxPtr;

    if (idx >= PROF_SLOTS_PER_CORE) {
        // Write overflow sentinel once (at last slot) for parser detection
        if (idx == PROF_SLOTS_PER_CORE) {
            __gm__ uint32_t* lastSlot = (__gm__ uint32_t*)(profBufGm + PROF_RINGIDX_REGION +
                coreId * PROF_SLOTS_PER_CORE * PROF_RECORD_BYTES +
                (PROF_SLOTS_PER_CORE - 1) * PROF_RECORD_BYTES);
            GmWrite32(lastSlot, EVT_OVERFLOW);
        }
        return;
    }

    uint64_t cycle = AscendC::GetSystemCycle();
    __gm__ uint32_t* slot = (__gm__ uint32_t*)(profBufGm + PROF_RINGIDX_REGION +
                                coreId * PROF_SLOTS_PER_CORE * PROF_RECORD_BYTES +
                                idx * PROF_RECORD_BYTES);

    // Write event record: [eventId, payload, cycle_lo, cycle_hi]
    GmWrite32(&slot[0], eventId);
    GmWrite32(&slot[1], payload);
    GmWrite64(&slot[2], cycle);

    // Increment ringIdx
    GmWrite32(ringIdxPtr, idx + 1);
}

} // namespace Profiler

#define PROF_INIT(buf)           Profiler::Init((__gm__ uint8_t*)(buf))
#define PROF_MARK(buf, evt, ...) Profiler::Mark((__gm__ uint8_t*)(buf), (evt), ##__VA_ARGS__)

// Maximum events buffered locally (UB-stack) per kernel instance before
// PROF_FINALIZE flushes them to the GM ring buffer. Distinct from
// PROF_SLOTS_PER_CORE (GM ring capacity). Bounded small to keep UB footprint
// predictable; raise only if a kernel legitimately emits >64 events.
#define PROF_MAX_EVENTS 64

#define PROF_LOCAL_DEFINE() \
    uint64_t _prof_timestamps[PROF_MAX_EVENTS]; \
    uint32_t _prof_event_ids[PROF_MAX_EVENTS]; \
    uint32_t _prof_count = 0; \
    __gm__ uint8_t* _prof_buf_gm = nullptr;

// C1 fix: pass the GlobalTensor object (_ringIdxGtRead / _flushGt) to
// DataCacheCleanAndInvalid, NOT _ringIdxGtRead[0] / _flushGt[0]. The gt[0]
// form in subpackage-path.md L160/L176 does not match GmWrite32/GmWrite64
// (which pass `gt`) and may not compile.
#define PROF_FINALIZE() \
    do { \
        if (_prof_buf_gm != nullptr && _prof_count > 0) { \
            uint32_t _coreId = Profiler::GetCoreId(); \
            if (_coreId < PROF_N_CORES) { \
                __gm__ uint32_t* _ringIdxPtr = (__gm__ uint32_t*)(_prof_buf_gm + _coreId * PROF_RINGIDX_ALIGN); \
                AscendC::GlobalTensor<uint32_t> _ringIdxGtRead; \
                _ringIdxGtRead.SetGlobalBuffer(_ringIdxPtr, 1); \
                AscendC::DataCacheCleanAndInvalid<uint32_t, AscendC::CacheLine::SINGLE_CACHE_LINE, AscendC::DcciDst::CACHELINE_OUT>(_ringIdxGtRead); \
                pipe_barrier(PIPE_ALL); \
                uint32_t _startIdx = *_ringIdxPtr; \
                if (_startIdx >= PROF_SLOTS_PER_CORE) _startIdx = 0; \
                if (_startIdx + _prof_count > PROF_SLOTS_PER_CORE) _prof_count = PROF_SLOTS_PER_CORE - _startIdx; \
                __gm__ uint32_t* _dstSlot = (__gm__ uint32_t*)(_prof_buf_gm + PROF_RINGIDX_REGION + \
                    _coreId * PROF_SLOTS_PER_CORE * PROF_RECORD_BYTES + _startIdx * PROF_RECORD_BYTES); \
                for (uint32_t _i = 0; _i < _prof_count; _i++) { \
                    Profiler::GmWrite32(&_dstSlot[_i*4],     _prof_event_ids[_i]); \
                    Profiler::GmWrite32(&_dstSlot[_i*4 + 1], 0); \
                    Profiler::GmWrite64(&_dstSlot[_i*4 + 2], _prof_timestamps[_i]); \
                } \
                pipe_barrier(PIPE_ALL); \
                Profiler::GmWrite32(_ringIdxPtr, _startIdx + _prof_count); \
                for (uint32_t _i = 0; _i < _prof_count * 4 + 1; _i++) { \
                    AscendC::GlobalTensor<uint32_t> _flushGt; \
                    _flushGt.SetGlobalBuffer(_dstSlot + _i, 1); \
                    AscendC::DataCacheCleanAndInvalid<uint32_t, AscendC::CacheLine::SINGLE_CACHE_LINE, AscendC::DcciDst::CACHELINE_OUT>(_flushGt); \
                } \
            } \
        } \
    } while (0)

// PROF_FINALIZE_AIC — AIC-specific: direct GM pointer writes, no TPipe dependency.
// Same flush logic as PROF_FINALIZE; GmWrite32/GmWrite64 construct a local
// GlobalTensor (not TBuf<>) for DCCI, so they are safe on AIC cores that lack
// TPipe/TBuf. Kept as a distinct macro so AIC call sites are grep-able and the
// no-TPipe intent is explicit at the call site.
#define PROF_FINALIZE_AIC() \
    do { \
        if (_prof_buf_gm != nullptr && _prof_count > 0) { \
            uint32_t _coreId = Profiler::GetCoreId(); \
            if (_coreId < PROF_N_CORES) { \
                __gm__ uint32_t* _ringIdxPtr = (__gm__ uint32_t*)(_prof_buf_gm + _coreId * PROF_RINGIDX_ALIGN); \
                AscendC::GlobalTensor<uint32_t> _ringIdxGtRead; \
                _ringIdxGtRead.SetGlobalBuffer(_ringIdxPtr, 1); \
                AscendC::DataCacheCleanAndInvalid<uint32_t, AscendC::CacheLine::SINGLE_CACHE_LINE, AscendC::DcciDst::CACHELINE_OUT>(_ringIdxGtRead); \
                pipe_barrier(PIPE_ALL); \
                uint32_t _startIdx = *_ringIdxPtr; \
                if (_startIdx >= PROF_SLOTS_PER_CORE) _startIdx = 0; \
                if (_startIdx + _prof_count > PROF_SLOTS_PER_CORE) _prof_count = PROF_SLOTS_PER_CORE - _startIdx; \
                __gm__ uint32_t* _dstSlot = (__gm__ uint32_t*)(_prof_buf_gm + PROF_RINGIDX_REGION + \
                    _coreId * PROF_SLOTS_PER_CORE * PROF_RECORD_BYTES + _startIdx * PROF_RECORD_BYTES); \
                for (uint32_t _i = 0; _i < _prof_count; _i++) { \
                    Profiler::GmWrite32(&_dstSlot[_i*4],     _prof_event_ids[_i]); \
                    Profiler::GmWrite32(&_dstSlot[_i*4 + 1], 0); \
                    Profiler::GmWrite64(&_dstSlot[_i*4 + 2], _prof_timestamps[_i]); \
                } \
                pipe_barrier(PIPE_ALL); \
                Profiler::GmWrite32(_ringIdxPtr, _startIdx + _prof_count); \
                for (uint32_t _i = 0; _i < _prof_count * 4 + 1; _i++) { \
                    AscendC::GlobalTensor<uint32_t> _flushGt; \
                    _flushGt.SetGlobalBuffer(_dstSlot + _i, 1); \
                    AscendC::DataCacheCleanAndInvalid<uint32_t, AscendC::CacheLine::SINGLE_CACHE_LINE, AscendC::DcciDst::CACHELINE_OUT>(_flushGt); \
                } \
            } \
        } \
    } while (0)

#else  // !ENABLE_PROFILING

#define PROF_INIT(buf)           ((void)0)
#define PROF_MARK(buf, evt, ...) ((void)0)
#define PROF_LOCAL_DEFINE()      ((void)0)
#define PROF_FINALIZE()          ((void)0)
#define PROF_FINALIZE_AIC()      ((void)0)

#endif
