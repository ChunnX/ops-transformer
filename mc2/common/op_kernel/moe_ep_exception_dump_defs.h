#ifndef MOE_EP_EXCEPTION_DUMP_DEFS_H
#define MOE_EP_EXCEPTION_DUMP_DEFS_H

#include <cstdint>

constexpr uint64_t MOE_EP_UB_ALIGN = 32;
constexpr uint64_t MOE_EP_WIN_ALIGN = 512;
constexpr uint64_t MOE_EP_NOTIFY_TOKEN_LIMIT = 15000;
constexpr uint64_t MOE_EP_MAX_OUT_DTYPE_SIZE = 2;
constexpr uint64_t MOE_EP_METADATA_DTYPE_SIZE = 4;

constexpr uint64_t MOE_EP_METADATA_SIZE = 64UL * 1024UL;
constexpr uint64_t MOE_EP_PER_CORE_DIAG_SLOTS = 100UL;
constexpr uint64_t MOE_EP_PER_CORE_DIAG_SLOT_SIZE = MOE_EP_WIN_ALIGN;
constexpr uint64_t MOE_EP_PER_CORE_DIAG_SIZE = MOE_EP_PER_CORE_DIAG_SLOTS * MOE_EP_PER_CORE_DIAG_SLOT_SIZE;
constexpr uint64_t MOE_EP_FIXED_PREFIX = MOE_EP_METADATA_SIZE + MOE_EP_PER_CORE_DIAG_SIZE;

constexpr uint32_t MOE_EP_DUMP_REGION_COUNT = 7U;

constexpr uint32_t MOE_EP_DUMP_REGION_PER_CORE_DIAG = 0U;
constexpr uint32_t MOE_EP_DUMP_REGION_COUNT_NOTIFY = 1U;
constexpr uint32_t MOE_EP_DUMP_REGION_EXPERT_COUNT = 2U;
constexpr uint32_t MOE_EP_DUMP_REGION_DISPATCH_SLOT_STATE = 3U;
constexpr uint32_t MOE_EP_DUMP_REGION_COMBINE_TOKEN_STATE = 4U;
constexpr uint32_t MOE_EP_DUMP_REGION_COMBINE_STATE_TAIL = 5U;
constexpr uint32_t MOE_EP_DUMP_REGION_HYBRID_SCALEOUT_STATUS = 6U;

constexpr uint32_t MOE_EP_DUMP_LAYOUT_VERSION = 1U;

constexpr uint32_t MOE_EP_NETWORK_DIRECT = 0U;
constexpr uint32_t MOE_EP_NETWORK_HYBRID = 1U;

struct MoeEpDumpRegion {
    uint64_t offset;
    uint64_t size;
};

struct MoeEpDumpMetadata {
    uint32_t layoutVersion;
    uint32_t nmt;
    uint32_t topK;
    uint32_t hidden;
    uint32_t epWorldSize;
    uint32_t localExpertNum;
    uint32_t aivNum;
    uint32_t networkMode;
    uint32_t serverNum;
    uint32_t rankNumPerServer;
    MoeEpDumpRegion regions[MOE_EP_DUMP_REGION_COUNT];
};

#endif
