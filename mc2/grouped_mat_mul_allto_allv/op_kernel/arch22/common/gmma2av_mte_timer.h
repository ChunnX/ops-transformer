#ifndef GMMA2AV_MTE_TIMER_H
#define GMMA2AV_MTE_TIMER_H
#include "kernel_operator.h"
namespace MC2KernelTemplate {
// Version 3 scalar timeline tracer, adapted to the Stage/Comm role split and
// the 0/1/2 round protocol. Recording adds no pipeline barrier: spans
// describe scalar issue/control flow, not asynchronous engine busy time.
// High-frequency facts (copy blocks, poll iterations, reset batches, MM
// tiles) are accumulated in per-lane counters flushed at Finish instead of
// per-iteration records. Buffer ownership is cache-line isolated; Finish
// publishes record counts and flushes every written line. Timer mode reuses
// mm_y and skips shared MM.
namespace GmmA2avMteTimerDetail {
constexpr int64_t HEADER_MAGIC = 0x474D4D4154494D45LL;
constexpr uint64_t CACHE_SLOTS = AscendC::CACHE_LINE_SIZE / sizeof(int64_t);
constexpr uint64_t HEADER_SLOTS = CACHE_SLOTS;
constexpr uint64_t LANE_META_SLOTS = 4UL;
constexpr uint64_t COUNTER_SLOTS = 16UL;
constexpr int64_t FORMAT_VERSION = 3LL;
constexpr int64_t LANE_COMPLETE = 0x454E444C414E45LL;
static_assert(HEADER_SLOTS >= 8UL, "Timer header needs a full cache line");
enum MteTimerEvent : int32_t {
    TIMER_EVENT_NONE = 0,
    TIMER_KERNEL_BEGIN = 1,
    TIMER_KERNEL_END = 2,
    TIMER_TILING_LOAD_END = 3,
    TIMER_PIPELINE_BEGIN = 4,
    TIMER_PIPELINE_END = 5,
    TIMER_RESET_SYNC_BEGIN = 6,
    TIMER_RESET_SYNC_END = 7,
    TIMER_COUNT_TABLE_BEGIN = 8,
    TIMER_COUNT_TABLE_END = 9,
    TIMER_COUNT_READY_BEGIN = 10,
    TIMER_COUNT_READY_END = 11,
    TIMER_BUILD_DEST_OFFSETS_BEGIN = 12,
    TIMER_BUILD_DEST_OFFSETS_END = 13,
    TIMER_AIC_KERNEL_BEGIN = 14,
    TIMER_AIC_KERNEL_END = 15,
    // payload = expertIdx; one owned M tile issue span, not cube busy.
    TIMER_AIC_MM_BEGIN = 16,
    TIMER_AIC_MM_END = 17,
    // payload = expertIdx; existing FIX_S wait plus wave-progress publication.
    TIMER_AIC_NOTIFY_BEGIN = 18,
    TIMER_AIC_NOTIFY_END = 19,
    TIMER_AIC_EXIT_WAIT_BEGIN = 20,
    TIMER_AIC_EXIT_WAIT_END = 21,
    // payload = wave sequence; Stage waits on its paired AIC progress.
    TIMER_C2V_WAIT_BEGIN = 22,
    TIMER_C2V_WAIT_END = 23,
    // payload = expertIdx; Stage workspace→window copy issue span.
    TIMER_STAGE_COPY_BEGIN = 24,
    TIMER_STAGE_COPY_END = 25,
    // payload = expertIdx; local progress publication.
    TIMER_STAGE_PROGRESS_BEGIN = 26,
    TIMER_STAGE_PROGRESS_END = 27,
    // payload = expertIdx; Comm wait of local producers' progress.
    TIMER_WAIT_STAGE_BEGIN = 28,
    TIMER_WAIT_STAGE_END = 29,
    // payload = expertIdx; ExpertReady publication to destinations.
    TIMER_PUBLISH_BEGIN = 30,
    TIMER_PUBLISH_END = 31,
    // payload = expertIdx; Comm wait of peer ExpertReady flags.
    TIMER_PEER_WAIT_BEGIN = 32,
    TIMER_PEER_WAIT_END = 33,
    TIMER_BUILD_SRC_OFFSETS_BEGIN = 34,
    TIMER_BUILD_SRC_OFFSETS_END = 35,
    // payload = expertIdx; Comm pull of peer payload.
    TIMER_PULL_BEGIN = 36,
    TIMER_PULL_END = 37,
    TIMER_RECV_DONE_BEGIN = 38,
    TIMER_RECV_DONE_END = 39,
    TIMER_EXIT_SYNC_BEGIN = 40,
    TIMER_EXIT_SYNC_END = 41,
};
// Per-lane counter slots, written after the four metadata slots at Finish.
enum MteTimerCounter : int32_t {
    CNT_STAGE_COPY_ISSUES = 0, // CopyGmToGmSegment loop iterations (Stage)
    CNT_STAGE_COPY_FULL = 1,   // blocks of exactly ubMoveElements_ elements
    CNT_STAGE_COPY_TAIL = 2,  // blocks smaller than ubMoveElements_ elements
    CNT_STAGE_COPY_BYTES = 3,
    CNT_PULL_COPY_ISSUES = 4, // CopyGmToGmSegment loop iterations (Comm pull)
    CNT_PULL_COPY_FULL = 5,
    CNT_PULL_COPY_TAIL = 6,
    CNT_PULL_COPY_BYTES = 7,
    CNT_STAGE_DRAINS = 8, // EndGmToGmCopy invocations on the Stage role
    CNT_PULL_DRAINS = 9,  // EndGmToGmCopy invocations on the Comm role
    CNT_POLLS = 10,       // flag-poll loop iterations (all wait helpers)
    CNT_RESET_BATCHES = 11,
    CNT_RESET_BYTES = 12,
    CNT_RESET_WRITES = 13,
    CNT_MM_TILES = 14,   // AIC: owned M×N tiles issued (experts with work)
    CNT_MM_EXPERTS = 15, // AIC: experts this core actually computed
};
} // namespace GmmA2avMteTimerDetail

class GmmA2avMteTimer {
public:
    __aicore__ inline void Init(GM_ADDR buffer, uint64_t bufferBytes, uint32_t laneId, uint32_t laneCount, bool enabled)
    {
        using namespace GmmA2avMteTimerDetail;
        enabled_ = enabled && buffer != 0UL && laneCount != 0U && laneId < laneCount &&
                   reinterpret_cast<uint64_t>(buffer) % AscendC::CACHE_LINE_SIZE == 0UL;
        if (!enabled_) {
            return;
        }
        const uint64_t totalSlots = bufferBytes / sizeof(int64_t);
        if (totalSlots <= HEADER_SLOTS) {
            enabled_ = false;
            return;
        }
        slotsPerLane_ = ((totalSlots - HEADER_SLOTS) / laneCount / CACHE_SLOTS) * CACHE_SLOTS;
        if (slotsPerLane_ < LANE_META_SLOTS + COUNTER_SLOTS + 2UL) {
            enabled_ = false;
            return;
        }
        lane_ = laneId;
        laneBase_ = HEADER_SLOTS + static_cast<uint64_t>(lane_) * slotsPerLane_;
        gmTimer_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(buffer));
        // Count, overflow, completion marker, lane identity. Counts delimit
        // this invocation, so no full-buffer zeroing or stale-tail heuristic.
        for (uint32_t i = 0U; i < LANE_META_SLOTS; ++i) {
            gmTimer_(laneBase_ + i) = 0;
        }
        for (uint32_t i = 0U; i < COUNTER_SLOTS; ++i) {
            gmTimer_(laneBase_ + LANE_META_SLOTS + i) = 0;
        }
        if (lane_ == 0U) {
            gmTimer_(0) = HEADER_MAGIC;
            gmTimer_(1) = static_cast<int64_t>(laneCount);
            gmTimer_(2) = static_cast<int64_t>(slotsPerLane_);
            gmTimer_(3) = FORMAT_VERSION;
            gmTimer_(4) = static_cast<int64_t>(HEADER_SLOTS);
            gmTimer_(5) = static_cast<int64_t>(LANE_META_SLOTS);
            gmTimer_(6) = static_cast<int64_t>(AscendC::CACHE_LINE_SIZE);
            gmTimer_(7) = 50LL; // GetSystemCycle ticks per microsecond on A3.
        }
    }

    __aicore__ inline void Record(int32_t event, uint32_t chunk)
    {
        if (!enabled_) {
            return;
        }
        if (GmmA2avMteTimerDetail::LANE_META_SLOTS + GmmA2avMteTimerDetail::COUNTER_SLOTS + writeIdx_ + 2UL >
            slotsPerLane_) {
            ++dropped_;
            return;
        }
        const int64_t cycle = AscendC::GetSystemCycle();
        WriteRecord(cycle, event, chunk);
    }

    // Record a span start whose beginning predates timer initialization
    // (kernel entry / tiling load): the caller captured the cycle itself.
    __aicore__ inline void RecordAt(int32_t event, uint32_t chunk, int64_t cycle)
    {
        if (!enabled_) {
            return;
        }
        if (GmmA2avMteTimerDetail::LANE_META_SLOTS + GmmA2avMteTimerDetail::COUNTER_SLOTS + writeIdx_ + 2UL >
            slotsPerLane_) {
            ++dropped_;
            return;
        }
        WriteRecord(cycle, event, chunk);
    }

    __aicore__ inline void AddCounter(int32_t counter, uint64_t value)
    {
        if (!enabled_) {
            return;
        }
        if (counter >= 0 && static_cast<uint64_t>(counter) < GmmA2avMteTimerDetail::COUNTER_SLOTS) {
            counters_[static_cast<uint64_t>(counter)] += value;
        }
    }

    __aicore__ inline void Finish()
    {
        if (!enabled_) {
            return;
        }
        using namespace GmmA2avMteTimerDetail;
        gmTimer_(laneBase_) = static_cast<int64_t>(writeIdx_ / 2UL);
        gmTimer_(laneBase_ + 1UL) = static_cast<int64_t>(dropped_);
        gmTimer_(laneBase_ + 2UL) = LANE_COMPLETE;
        gmTimer_(laneBase_ + 3UL) = static_cast<int64_t>(lane_);
        for (uint32_t i = 0U; i < COUNTER_SLOTS; ++i) {
            gmTimer_(laneBase_ + LANE_META_SLOTS + i) = static_cast<int64_t>(counters_[i]);
        }
        for (uint64_t offset = 0UL; offset < LANE_META_SLOTS + COUNTER_SLOTS + writeIdx_; offset += CACHE_SLOTS) {
            AscendC::DataCacheCleanAndInvalid<int64_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
                                              AscendC::DcciDst::CACHELINE_OUT>(gmTimer_[laneBase_ + offset]);
        }
        if (lane_ == 0U) {
            AscendC::DataCacheCleanAndInvalid<int64_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
                                              AscendC::DcciDst::CACHELINE_OUT>(gmTimer_);
        }
        AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
    }

    __aicore__ inline bool Enabled() const
    {
        return enabled_;
    }

private:
    __aicore__ inline void WriteRecord(int64_t cycle, int32_t event, uint32_t chunk)
    {
        const uint64_t index = laneBase_ + GmmA2avMteTimerDetail::LANE_META_SLOTS +
                               GmmA2avMteTimerDetail::COUNTER_SLOTS + writeIdx_;
        gmTimer_(index) = cycle;
        gmTimer_(index + 1) = (static_cast<int64_t>(chunk) << 16) | static_cast<int64_t>(event);
        writeIdx_ += 2UL;
    }

    AscendC::GlobalTensor<int64_t> gmTimer_;
    uint64_t slotsPerLane_{0UL};
    uint64_t laneBase_{0UL};
    uint64_t writeIdx_{0UL};
    uint64_t dropped_{0UL};
    uint64_t counters_[GmmA2avMteTimerDetail::COUNTER_SLOTS]{};
    uint32_t lane_{0U};
    bool enabled_{false};
};

// The entry dispatches once on the timer flag. This type removes every
// record/count/flush operation from the untraced hot loops at compile time.
struct GmmA2avMteNullTimer {
    __aicore__ inline void Init(GM_ADDR, uint64_t, uint32_t, uint32_t, bool) {}
    __aicore__ inline void Record(int32_t, uint32_t) {}
    __aicore__ inline void RecordAt(int32_t, uint32_t, int64_t) {}
    __aicore__ inline void AddCounter(int32_t, uint64_t) {}
    __aicore__ inline void Finish() {}
    __aicore__ inline bool Enabled() const { return false; }
};


// Scope brackets add timestamps only, including early-return paths.
template <typename TimerType>
struct GmmA2avMteTimerScope {
    TimerType &timer;
    int32_t event;
    uint32_t payload;
    __aicore__ inline GmmA2avMteTimerScope(TimerType &t, int32_t e, uint32_t p)
        : timer(t), event(e), payload(p) { timer.Record(event, payload); }
    __aicore__ inline ~GmmA2avMteTimerScope() { timer.Record(event + 1, payload); }
};
} // namespace MC2KernelTemplate
#endif
