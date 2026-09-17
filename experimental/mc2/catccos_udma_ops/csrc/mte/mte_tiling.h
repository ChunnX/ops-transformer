// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// Licensed under CANN Open Software License Agreement Version 2.0.
#pragma once

#include <c10/util/Exception.h>
#include "catccos/comm/mte_transport_params.hpp"

namespace OpsTransformer::CatccosMte {

inline MteCommConfig ResolveMteCommConfig(
    MteCommConfig config, uint32_t rankSize, uint32_t availableRemoteAivs)
{
    TORCH_CHECK(rankSize >= 1 && rankSize <= kMteMaxRanks, "MTE supports 1 to 8 ranks");
    TORCH_CHECK(config.workersPerPeer >= 1 && config.workersPerPeer <= kMteMaxPeerWorkers,
                "workersPerPeer must be in [1, 64]");
    TORCH_CHECK(config.commUbStages >= 1 && config.commUbStages <= kMteMaxUbStages,
                "commUbStages must be 1 or 2");
    TORCH_CHECK(config.commChunkBytes >= 32 && config.commChunkBytes % 32 == 0,
                "commChunkBytes must be a positive multiple of 32 bytes");
    const uint64_t ubBytes = uint64_t{config.commChunkBytes} * config.commUbStages +
                            kMteSignalScratchBytes + kMteSystemReservedBytes;
    TORCH_CHECK(ubBytes <= kMtePhysicalUbBytes,
                "MTE UB budget exceeded: payload + control + reserved = ", ubBytes,
                " bytes, capacity = ", kMtePhysicalUbBytes);
    if (rankSize == 1) {
        TORCH_CHECK(config.commAivCount == 0 && config.workersPerPeer == 1,
                    "single-rank MTE requires commAivCount=0 and workersPerPeer=1");
    } else {
        TORCH_CHECK(config.commAivCount >= config.workersPerPeer &&
                    config.commAivCount <= availableRemoteAivs &&
                    config.commAivCount % config.workersPerPeer == 0,
                    "commAivCount must fit the remote AIV pool and be a positive multiple of workersPerPeer; available=",
                    availableRemoteAivs);
        TORCH_CHECK(config.commAivCount / config.workersPerPeer <= rankSize - 1,
                    "MTE worker groups must not exceed the number of remote peers");
    }
    return config;
}

} // namespace OpsTransformer::CatccosMte
