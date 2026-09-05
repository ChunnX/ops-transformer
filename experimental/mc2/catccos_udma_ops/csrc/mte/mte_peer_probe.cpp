// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// Licensed under CANN Open Software License Agreement Version 2.0.

#include "mte_runtime.h"
#include "kernel_operator.h"
#include "catlass/arch/arch.hpp"
#include "catlass/arch/resource.hpp"
#include "catccos/comm/block/comm_block_mte.hpp"
#include "shmem.h"
#include "shmemi_device_common.h"

// This initialization-only control-plane dependency is deliberately tied to
// the pinned SHMEM source. Public host barriers cannot exchange failure state,
// and device barriers would depend on the very links this probe is checking.
#include "utils/shmemi_host_types.h"

#include <array>
#include <cstdint>
#include <sstream>
#include <stdexcept>

extern aclshmemi_bootstrap_handle_t g_boot_handle;

namespace OpsTransformer::CatccosMte {
namespace {

constexpr uint32_t kProbeBytes = 4096;
constexpr uint32_t kSeedOffset = 256;
constexpr uint32_t kGetOffset = 1024;
constexpr uint32_t kPutOffset = 2048;
constexpr uint32_t kPeerBytes = 64;
constexpr uint32_t kPeerWords = kPeerBytes / sizeof(uint32_t);
constexpr uint32_t kProbeMagic = 0x4d544531;
constexpr int32_t kProbeSyncTimeoutMs = 30000;
static_assert(kPutOffset + kMteMaxRanks * kPeerBytes <= kProbeBytes);
static_assert(kProbeBytes <= MteRuntime::kOperatorControlBytes);

struct PhaseStatus {
    int32_t rank{0};
    int32_t aclStatus{0};
    uint32_t unsafeToRelease{0};
    uint32_t invalidDeviceState{0};
    uint32_t missingMtePeers{0};
    uint32_t missingMappings{0};
    uint32_t mismatchedPeers{0};
    uint32_t firstWord{0};
    uint32_t expected{0};
    uint32_t observed{0};
};

uint32_t Pattern(uint32_t source, uint32_t destination, uint32_t word)
{
    return 0x6d700000U | (source << 12) | (destination << 8) | word;
}

// This kernel reads only local SHMEM state; no remote access or device
// collective can occur before all hosts have accepted every directed edge.
extern "C" ACLSHMEM_GLOBAL_VECTOR void MtePeerTopologyProbe(
    GM_ADDR scratch, int32_t rank, int32_t worldSize)
{
    if (AscendC::GetBlockIdx() != 0) {
        return;
    }
    auto result = reinterpret_cast<__gm__ uint32_t *>(scratch);
    auto state = aclshmemi_get_state();
    uint32_t invalidState = state == nullptr;
    uint32_t missingMte = 0;
    uint32_t missingMappings = 0;
    if (state != nullptr) {
        invalidState = state->mype != rank || state->npes != worldSize || state->heap_base == nullptr;
        for (int32_t peer = 0; peer < worldSize; ++peer) {
            if ((state->topo_list[peer] & ACLSHMEM_TRANSPORT_MTE) == 0) {
                missingMte |= uint32_t{1} << peer;
            }
            // A translated pointer alone is insufficient: aclshmem_ptr adds
            // the offset even when the peer base address is unavailable.
            if (state->p2p_device_heap_base[peer] == nullptr ||
                aclshmem_ptr(scratch, peer) == nullptr) {
                missingMappings |= uint32_t{1} << peer;
            }
        }
    }
    result[0] = kProbeMagic;
    result[1] = invalidState;
    result[2] = missingMte;
    result[3] = missingMappings;
    dcci_cacheline(scratch);
}

template <Catccos::Comm::MteCopyDirection Direction>
ACLSHMEM_GLOBAL_VECTOR void MtePeerDataProbe(
    GM_ADDR scratch, int32_t rank, int32_t worldSize, uint64_t fftsAddress)
{
    if (AscendC::GetBlockIdx() != 0) {
        return;
    }
    if (fftsAddress != 0) {
        util_set_ffts_config(fftsAddress);
    }
    using Block = Catccos::Comm::Block::MteCommBlock<uint32_t, Direction>;
    Catlass::Arch::Resource<Catlass::Arch::Ascend950> resource;
    Catlass::MatrixCoord shape{1, kPeerWords};
    typename Block::TileRemoteCopy::Params tile{shape};
    typename Block::Params params{shape, tile, MteCommConfig{0, 1, kPeerBytes, 1}};
    Block copy(resource, params);
    copy.SetWorker(0, 1);
    copy.InitBlockLoop();
    auto layout = Catlass::layout::RowMajor(1, kPeerWords);
    for (int32_t peer = 0; peer < worldSize; ++peer) {
        AscendC::GlobalTensor<uint32_t> source;
        AscendC::GlobalTensor<uint32_t> destination;
        if constexpr (Direction == Catccos::Comm::MteCopyDirection::Get) {
            // Read the remote seed addressed to this rank into its peer slot.
            source.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t *>(
                scratch + kSeedOffset + rank * kPeerBytes));
            destination.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t *>(
                scratch + kGetOffset + peer * kPeerBytes));
        } else {
            // Every source owns a disjoint slot in every destination's inbox.
            source.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t *>(
                scratch + kSeedOffset + peer * kPeerBytes));
            destination.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t *>(
                scratch + kPutOffset + rank * kPeerBytes));
        }
        copy(source, layout, destination, layout, shape, peer);
    }
    copy.FinalizeBlockLoop();
    // Drain before kernel completion; scratch/event 7 used by SHMEM signaling
    // is never used by this probe. The common block owns payload event 0.
    AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(kMteCompletionEvent);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(kMteCompletionEvent);
}

void CheckEveryRank(const PhaseStatus &local, int32_t worldSize, const char *phase)
{
    if (g_boot_handle.allgather == nullptr) {
        throw MteProbeError("MTE initialization requires the pinned SHMEM bootstrap allgather", true);
    }
    std::array<PhaseStatus, kMteMaxRanks> gathered{};
    const int status = g_boot_handle.allgather(
        &local, gathered.data(), static_cast<int>(sizeof(local)), &g_boot_handle);
    if (status != 0) {
        std::ostringstream message;
        message << "MTE probe bootstrap allgather failed during " << phase << ": " << status;
        throw MteProbeError(message.str(), true);
    }
    bool failed = false;
    bool unsafe = false;
    std::ostringstream details;
    for (int32_t rank = 0; rank < worldSize; ++rank) {
        const auto &result = gathered[rank];
        unsafe |= result.unsafeToRelease != 0;
        if (result.rank != rank || result.aclStatus != ACL_SUCCESS || result.invalidDeviceState ||
            result.missingMtePeers || result.missingMappings || result.mismatchedPeers) {
            failed = true;
            details << " [rank=" << rank << ", acl=" << result.aclStatus
                    << ", invalid_state=" << result.invalidDeviceState
                    << ", missing_mte_mask=" << result.missingMtePeers
                    << ", missing_mapping_mask=" << result.missingMappings
                    << ", mismatch_mask=" << result.mismatchedPeers
                    << ", word=" << result.firstWord << ", expected=" << result.expected
                    << ", observed=" << result.observed << ']';
        }
    }
    if (failed) {
        const std::string message = std::string("MTE peer probe failed during ") + phase + details.str();
        // All ranks make the same cleanup decision. A timed-out kernel may
        // still access scratch, so the runtime must quarantine this context.
        if (unsafe) {
            throw MteProbeError(message, true);
        }
        throw std::runtime_error(message);
    }
}

PhaseStatus AwaitKernel(aclrtStream stream, int32_t rank)
{
    PhaseStatus result;
    result.rank = rank;
    result.aclStatus = aclrtSynchronizeStreamWithTimeout(stream, kProbeSyncTimeoutMs);
    result.unsafeToRelease = result.aclStatus != ACL_SUCCESS;
    return result;
}

void CheckPayload(const uint32_t *words, int32_t rank, int32_t worldSize, PhaseStatus &status)
{
    for (int32_t peer = 0; peer < worldSize; ++peer) {
        for (uint32_t word = 0; word < kPeerWords; ++word) {
            const uint32_t expected = Pattern(peer, rank, word);
            const uint32_t observed = words[peer * kPeerWords + word];
            if (observed != expected) {
                if (status.mismatchedPeers == 0) {
                    status.firstWord = word;
                    status.expected = expected;
                    status.observed = observed;
                }
                status.mismatchedPeers |= uint32_t{1} << peer;
            }
        }
    }
}

} // namespace

void ValidateMtePeers(uint8_t *workspace, int32_t rank, int32_t worldSize,
                      uint64_t fftsAddress, aclrtStream stream)
{
    // The caller passes the first control slice, not the 400-MiB data plane.
    // Initialization has exclusive ownership; success restores every byte used.
    std::array<uint32_t, kProbeBytes / sizeof(uint32_t)> initial{};
    for (uint32_t &word : initial) {
        word = 0xa5a5a5a5U;
    }
    for (int32_t peer = 0; peer < worldSize; ++peer) {
        for (uint32_t word = 0; word < kPeerWords; ++word) {
            initial[kSeedOffset / sizeof(uint32_t) + peer * kPeerWords + word] = Pattern(rank, peer, word);
        }
    }
    PhaseStatus status;
    status.rank = rank;
    status.aclStatus = aclrtMemcpy(workspace, kProbeBytes, initial.data(), kProbeBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    status.unsafeToRelease = status.aclStatus != ACL_SUCCESS;
    status.invalidDeviceState = g_boot_handle.mype != rank || g_boot_handle.npes != worldSize;
    CheckEveryRank(status, worldSize, "seed initialization");

    MtePeerTopologyProbe<<<1, nullptr, stream>>>(workspace, rank, worldSize);
    status = AwaitKernel(stream, rank);
    std::array<uint32_t, kPeerWords> topology{};
    if (status.aclStatus == ACL_SUCCESS) {
        status.aclStatus = aclrtMemcpy(topology.data(), kPeerBytes, workspace, kPeerBytes, ACL_MEMCPY_DEVICE_TO_HOST);
        status.unsafeToRelease = status.aclStatus != ACL_SUCCESS;
        if (status.aclStatus == ACL_SUCCESS) {
            status.invalidDeviceState = topology[0] != kProbeMagic || topology[1] != 0;
            status.missingMtePeers = topology[2];
            status.missingMappings = topology[3];
        }
    }
    CheckEveryRank(status, worldSize, "directed topology capability");

    MtePeerDataProbe<Catccos::Comm::MteCopyDirection::Get><<<1, nullptr, stream>>>(
        workspace, rank, worldSize, fftsAddress);
    status = AwaitKernel(stream, rank);
    std::array<uint32_t, kMteMaxRanks * kPeerWords> result{};
    if (status.aclStatus == ACL_SUCCESS) {
        status.aclStatus = aclrtMemcpy(result.data(), sizeof(result), workspace + kGetOffset,
                                       sizeof(result), ACL_MEMCPY_DEVICE_TO_HOST);
        status.unsafeToRelease = status.aclStatus != ACL_SUCCESS;
        if (status.aclStatus == ACL_SUCCESS) {
            CheckPayload(result.data(), rank, worldSize, status);
        }
    }
    CheckEveryRank(status, worldSize, "all-peer MTE GET");

    MtePeerDataProbe<Catccos::Comm::MteCopyDirection::Put><<<1, nullptr, stream>>>(
        workspace, rank, worldSize, fftsAddress);
    status = AwaitKernel(stream, rank);
    // A local stream completion alone does not prove that all other sources
    // have finished PUT. The host control collective joins every sender first.
    CheckEveryRank(status, worldSize, "all-peer MTE PUT completion");
    status.aclStatus = aclrtMemcpy(result.data(), sizeof(result), workspace + kPutOffset,
                                   sizeof(result), ACL_MEMCPY_DEVICE_TO_HOST);
    status.unsafeToRelease = status.aclStatus != ACL_SUCCESS;
    if (status.aclStatus == ACL_SUCCESS) {
        CheckPayload(result.data(), rank, worldSize, status);
    }
    CheckEveryRank(status, worldSize, "all-peer MTE PUT values");

    status.aclStatus = aclrtMemset(workspace, kProbeBytes, 0, kProbeBytes);
    status.unsafeToRelease = status.aclStatus != ACL_SUCCESS;
    CheckEveryRank(status, worldSize, "probe scratch cleanup");
}

} // namespace OpsTransformer::CatccosMte
