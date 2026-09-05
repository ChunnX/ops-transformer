// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// Licensed under CANN Open Software License Agreement Version 2.0.
#pragma once

#include <cstdint>

namespace OpsTransformer::CatccosMte {

// The four fields are part of the host/device tuning ABI, in this order.
// Counts refer to remote communication AIVs; self and export have separate pools.
struct MteCommConfig {
    uint32_t commAivCount{0};
    uint32_t workersPerPeer{1};
    uint32_t commChunkBytes{64 * 1024};
    uint32_t commUbStages{2};
};

inline constexpr uint32_t kMtePhysicalUbBytes = 256 * 1024;
inline constexpr uint32_t kMteSystemReservedBytes = 8 * 1024;
inline constexpr uint32_t kMteSignalScratchBytes = 2 * 1024;
inline constexpr uint32_t kMteMaxUbStages = 2;
inline constexpr uint32_t kMteMaxPeerWorkers = 64;
inline constexpr uint32_t kMteMaxRanks = 8;
inline constexpr uint32_t kMteCompletionEvent = 6;
inline constexpr uint32_t kMteSignalEvent = 7;
inline constexpr uint64_t kMteOperatorControlBytes = 64 * 1024;
inline constexpr uint64_t kMteMmadSignalOffset = 0;
inline constexpr uint64_t kMteCommSignalOffset = 64;
inline constexpr uint64_t kMteWorkerFlagsOffset = 256;
inline constexpr uint32_t kMteWorkerFlagStride = 64;

static_assert(sizeof(MteCommConfig) == 4 * sizeof(uint32_t));
static_assert(kMteWorkerFlagsOffset + uint64_t{kMteMaxRanks} * kMteMaxPeerWorkers *
                  kMteWorkerFlagStride <= kMteOperatorControlBytes);

} // namespace OpsTransformer::CatccosMte
