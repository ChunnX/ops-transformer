/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 * See LICENSE in the root of the software repository.
 */
#ifndef MOE_FULL_MESH_COC_TIMER_HPP
#define MOE_FULL_MESH_COC_TIMER_HPP

// Compact, AIV-only CocTimer adapter. See README.md for the PR provenance and ABI.
// Enable explicitly for diagnostic builds; the disabled kernel has no timer state.
#ifndef MOE_DISPATCH_FULL_MESH_TIMER
#define MOE_DISPATCH_FULL_MESH_TIMER 0
#endif

#ifndef MOE_DISPATCH_FULL_MESH_ARRIVAL_ONLY
#define MOE_DISPATCH_FULL_MESH_ARRIVAL_ONLY 0
#endif

#if MOE_DISPATCH_FULL_MESH_TIMER
#define MOE_FULL_MESH_TIMER(code) \
    do { \
        code; \
    } while (0)
namespace Mc2Kernel {
class MoeFullMeshCocTimer {
public:
    // ABI v6: flat serial stages per core, no containment. Send cores record
    // INIT -> CNT_SEND -> PAYLOAD -> RECV_COPY -> TAIL; cumsum cores record
    // INIT -> CNT_CALC -> RECV_COPY -> TAIL. Each stage starts at or after the
    // previous stage ends, so durations add up to the full kernel span.
    enum Event : uint32_t {
        INIT,
        CNT_SEND,
        PAYLOAD,
        CNT_CALC,
        RECV_COPY,
        TAIL,
        EVENT_COUNT
    };
    static constexpr uint64_t MAGIC = 0x434F434D4F453031ULL;
    static constexpr uint32_t HEADER_WORDS = 16;
    static constexpr uint32_t RECORD_WORDS = 64; // 512 B, four exclusive cache lines per AIV
    static_assert(8 + EVENT_COUNT * 2 <= RECORD_WORDS, "Update the dump ABI when adding events");

    // Callable at the very top of the kernel Init: only needs the tiling fields
    // available there. sendCores/tokenBytes/payloadBytes follow via SetMeta.
    __aicore__ inline void Init(GM_ADDR assist, uint64_t capacityA, uint32_t coreId, uint32_t coreCount, uint32_t rank)
    {
        coreId_ = coreId;
        for (uint32_t i = 0; i < EVENT_COUNT * 2; ++i) {
            cycles_[i] = 0;
        }
        // Dispatch/Combine use the first 3*A int32 values. Never share their last cache line.
        uint64_t offset = ((capacityA * 3 * sizeof(int32_t) + 127) / 128) * 128;
        uint64_t bytes = capacityA * 128 * sizeof(int32_t);
        uint64_t required = (HEADER_WORDS + uint64_t(coreCount) * RECORD_WORDS) * sizeof(uint64_t);
        enabled_ = assist != nullptr && offset + required <= bytes && coreId < coreCount;
        if (assist == nullptr || offset + HEADER_WORDS * sizeof(uint64_t) > bytes) {
            return;
        }
        gm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t *>(assist + offset));
        if (coreId == 0) {
            // status: 1 = enabled; 2 = insufficient tail space. Host reads after stream synchronization.
            uint64_t header[HEADER_WORDS] = {MAGIC,
                                             6,
                                             enabled_ ? 1ULL : 2ULL,
                                             capacityA,
                                             coreCount,
                                             RECORD_WORDS,
                                             EVENT_COUNT,
                                             rank,
                                             0,
                                             0,
                                             0,
                                             offset,
                                             required,
                                             50,
                                             1,
                                             MOE_DISPATCH_FULL_MESH_ARRIVAL_ONLY};
            for (uint32_t i = 0; i < HEADER_WORDS; ++i) {
                gm_.SetValue(i, header[i]);
            }
            AscendC::DataCacheCleanAndInvalid<uint64_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
                                              AscendC::DcciDst::CACHELINE_OUT>(gm_);
        }
    }

    // Fill the header metadata that is only known after the tiling derivation.
    __aicore__ inline void SetMeta(uint64_t sendCores, uint64_t tokenBytes, uint64_t payloadBytes)
    {
        sendCores_ = sendCores;
        tokenBytes_ = tokenBytes;
        payloadBytes_ = payloadBytes;
    }

    // Deliberately no PIPE_ALL here: preserve the communication pipeline and record observation times.
    __aicore__ inline void Tik(Event event)
    {
        if (enabled_ && cycles_[2 * event] == 0) {
            cycles_[2 * event] = static_cast<uint64_t>(AscendC::GetSystemCycle());
        }
    }
    // Close the interval opened by Tik; keep the earliest end for idempotent callers.
    __aicore__ inline void Tok(Event event)
    {
        if (enabled_ && cycles_[2 * event] != 0 && cycles_[2 * event + 1] == 0) {
            cycles_[2 * event + 1] = static_cast<uint64_t>(AscendC::GetSystemCycle());
        }
    }
    __aicore__ inline void Flush()
    {
        if (!enabled_) {
            return;
        }
        if (coreId_ == 0) {
            gm_.SetValue(8, sendCores_);
            gm_.SetValue(9, tokenBytes_);
            gm_.SetValue(10, payloadBytes_);
            AscendC::DataCacheCleanAndInvalid<uint64_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
                                              AscendC::DcciDst::CACHELINE_OUT>(gm_);
        }
        uint64_t base = HEADER_WORDS + uint64_t(coreId_) * RECORD_WORDS;
        // Write every slot, including unused events, so repeated launches cannot retain stale timestamps.
        uint64_t metadata[8] = {MAGIC, coreId_, 0, 0, 0, 0, 0, 0};
        for (uint32_t i = 0; i < 8; ++i) {
            gm_.SetValue(base + i, metadata[i]);
        }
        for (uint32_t i = 0; i < EVENT_COUNT * 2; ++i) {
            gm_.SetValue(base + 8 + i, cycles_[i]);
        }
        for (uint32_t i = 8 + EVENT_COUNT * 2; i < RECORD_WORDS; ++i) {
            gm_.SetValue(base + i, 0);
        }
        for (uint32_t i = 0; i < RECORD_WORDS; i += 16) {
            AscendC::DataCacheCleanAndInvalid<uint64_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
                                              AscendC::DcciDst::CACHELINE_OUT>(gm_[base + i]);
        }
    }

private:
    bool enabled_{false};
    uint32_t coreId_{0};
    uint64_t sendCores_{0};
    uint64_t tokenBytes_{0};
    uint64_t payloadBytes_{0};
    uint64_t cycles_[EVENT_COUNT * 2]{};
    AscendC::GlobalTensor<uint64_t> gm_;
};
} // namespace Mc2Kernel
#else
#define MOE_FULL_MESH_TIMER(code) \
    do { \
    } while (0)
#endif
#endif
