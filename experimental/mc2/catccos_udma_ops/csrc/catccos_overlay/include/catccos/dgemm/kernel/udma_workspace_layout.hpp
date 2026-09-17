/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */

#ifndef OPS_TRANSFORMER_CATCCOS_UDMA_WORKSPACE_LAYOUT_HPP
#define OPS_TRANSFORMER_CATCCOS_UDMA_WORKSPACE_LAYOUT_HPP

#include <cstdint>

namespace OpsTransformer::CatccosUdma {

// One symmetric allocation is shared by both fused operators. Keep all
// communication payloads in the data plane and reserve a disjoint tail for
// synchronization state. Each operator owns a separate control-plane slice so
// a faster rank entering the next fused operator cannot overwrite signals that
// a slower rank is still consuming in the previous one.
struct UdmaWorkspaceLayout {
    static constexpr uint64_t kDataBytes = 640UL * 1024 * 1024;
    static constexpr uint64_t kControlBase = 700UL * 1024 * 1024;
    static constexpr uint64_t kControlBytes = 4096;
    static constexpr uint64_t kTotalBytes = kControlBase + kControlBytes;

    static constexpr uint64_t kControlAlignment = 64;
    static constexpr uint32_t kMaxRankSize = 8;
    static constexpr uint64_t kRankSignalBytes =
        kMaxRankSize * sizeof(int32_t);

    // grouped_mat_mul_all_to_allv owns the first half of the control plane.
    static constexpr uint64_t kGmmA2avControlOffset = kControlBase;
    static constexpr uint64_t kGmmA2avControlBytes = 2048;
    static constexpr uint64_t kGmmA2avSyncMmadOffset =
        kGmmA2avControlOffset;
    static constexpr uint64_t kGmmA2avSyncCommOffset =
        kGmmA2avControlOffset + kControlAlignment;

    // all_to_allv_grouped_mat_mul owns the second half. Strong-ready uses one
    // uint64 slot for each [workspace stage, source rank].
    static constexpr uint64_t kA2avGmmControlOffset =
        kGmmA2avControlOffset + kGmmA2avControlBytes;
    static constexpr uint64_t kA2avGmmControlBytes = 2048;
    static constexpr uint64_t kA2avGmmSyncMmadOffset =
        kA2avGmmControlOffset;
    static constexpr uint64_t kA2avGmmReadyOffset =
        kA2avGmmControlOffset + 2 * kControlAlignment;
    static constexpr uint32_t kA2avGmmReadyStages = 3;
    static constexpr uint64_t kA2avGmmReadyBytes =
        kA2avGmmReadyStages * kMaxRankSize * sizeof(uint64_t);
};

static_assert(
    UdmaWorkspaceLayout::kDataBytes <= UdmaWorkspaceLayout::kControlBase,
    "UDMA controls overlap the symmetric data plane");
static_assert(
    UdmaWorkspaceLayout::kGmmA2avSyncCommOffset +
            UdmaWorkspaceLayout::kRankSignalBytes <=
        UdmaWorkspaceLayout::kGmmA2avControlOffset +
            UdmaWorkspaceLayout::kGmmA2avControlBytes,
    "grouped-matmul AllToAllV controls exceed their reserved slice");
static_assert(
    UdmaWorkspaceLayout::kA2avGmmReadyOffset +
            UdmaWorkspaceLayout::kA2avGmmReadyBytes <=
        UdmaWorkspaceLayout::kA2avGmmControlOffset +
            UdmaWorkspaceLayout::kA2avGmmControlBytes,
    "AllToAllV grouped-matmul controls exceed their reserved slice");
static_assert(
    UdmaWorkspaceLayout::kA2avGmmControlOffset +
            UdmaWorkspaceLayout::kA2avGmmControlBytes <=
        UdmaWorkspaceLayout::kTotalBytes,
    "UDMA controls exceed the symmetric workspace");
static_assert(
    UdmaWorkspaceLayout::kGmmA2avControlOffset %
                UdmaWorkspaceLayout::kControlAlignment ==
            0 &&
        UdmaWorkspaceLayout::kControlBase %
                UdmaWorkspaceLayout::kControlAlignment ==
            0 &&
        UdmaWorkspaceLayout::kA2avGmmControlOffset %
                UdmaWorkspaceLayout::kControlAlignment ==
            0,
    "UDMA control slices must be cache-line aligned");

}  // namespace OpsTransformer::CatccosUdma

#endif  // OPS_TRANSFORMER_CATCCOS_UDMA_WORKSPACE_LAYOUT_HPP
